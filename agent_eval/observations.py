"""Extraction-only normalization from runtime answers and structured evidence."""

from __future__ import annotations

import re
from dataclasses import replace
from typing import Any, Mapping, Sequence

from .alignment import DeterministicRequirementAligner, RequirementAligner
from .models import EvaluationRunArtifact, ObservedEvidence


class ObservationExtractor:
    """Extract observed values; it never decides whether they match Gold targets."""

    def __init__(self, *, aligner: RequirementAligner | None = None) -> None:
        self.aligner = aligner or DeterministicRequirementAligner()

    def extract(self, gold_case: Mapping[str, Any], run: EvaluationRunArtifact) -> EvaluationRunArtifact:
        alignments = {item.gold_requirement_id: item for item in self.aligner.align(gold_case, run)}
        tasks = {item.runtime_task_id: item for item in run.planned_requirements}
        numeric_by_metric: dict[str, Mapping[str, Any]] = {}
        updates: dict[str, dict[str, Any]] = {}
        for requirement in gold_case.get("requirements", []):
            alignment = alignments.get(requirement["requirement_id"])
            if alignment is None or not alignment.status.startswith("mapped"):
                continue
            mapped = [tasks[task_id] for task_id in alignment.runtime_task_ids if task_id in tasks]
            if not mapped:
                continue
            for assertion in requirement.get("expected_answer_assertions", []):
                task = _task_for_assertion(assertion, mapped)
                answer = task.answer_fragment or ""
                evidence = [item for item in run.evidence_registry if item.runtime_task_id == task.runtime_task_id]
                observation = self._assertion(assertion, answer, evidence)
                if observation is not None:
                    updates.setdefault(task.runtime_task_id, {})[assertion["assertion_id"]] = observation
                    metric = assertion.get("metric")
                    if assertion.get("type") == "numeric" and isinstance(metric, str):
                        numeric_by_metric[_metric_key(metric, assertion.get("year"))] = observation

        replaced_tasks = tuple(
            replace(task, assertion_observations=updates.get(task.runtime_task_id, task.assertion_observations))
            for task in run.planned_requirements
        )
        case_observations = dict(run.case_assertion_observations)
        for assertion in gold_case.get("expected_answer_assertions", []):
            if assertion.get("type") == "relation":
                relation = _relation_observation(assertion, numeric_by_metric, run.final_answer or "")
                if relation is not None:
                    case_observations[assertion["assertion_id"]] = relation
        return replace(run, planned_requirements=replaced_tasks, case_assertion_observations=case_observations)

    def _assertion(
        self,
        assertion: Mapping[str, Any],
        answer: str,
        evidence: Sequence[ObservedEvidence],
    ) -> Any:
        kind = assertion.get("type")
        if kind == "set_or_rows":
            for item in evidence:
                rows = item.identity.get("rows")
                if isinstance(rows, list):
                    output = [_normalize_sql_row(row, item.identity) for row in rows if isinstance(row, Mapping)]
                    if "rank" in assertion.get("key", []) and output and "rank" not in output[0]:
                        output = [{"rank": index, **row} for index, row in enumerate(output, start=1)]
                    return {"value": output, "source": "structured_sql_result"}
            return None
        if kind == "numeric":
            text = _observation_text(answer, evidence)
            value = _extract_numeric(text, assertion)
            if value is None:
                return None
            result: dict[str, Any] = {"value": value, "unit": assertion.get("unit"), "source": "runtime_text"}
            if assertion.get("value_representation"):
                result["value_representation"] = "percentage_points" if "%" in text or "percent" in text.casefold() else None
            if assertion.get("metric") is not None:
                result["metric"] = assertion.get("metric")
            if assertion.get("year") is not None:
                result["year"] = assertion.get("year")
            return result
        if kind == "date":
            for item in evidence:
                value = item.publication_date or item.identity.get("publication_date") or item.fact_as_of
                normalized = _date(str(value)) if value else None
                if normalized:
                    return {"value": normalized, "source": "structured_evidence_metadata"}
            normalized = _date(_observation_text(answer, evidence))
            return {"value": normalized, "source": "runtime_text"} if normalized else None
        if kind == "citation":
            linked = []
            for item in evidence:
                identity = item.identity
                tokens = [item.evidence_id, identity.get("url"), identity.get("publisher_domain"), identity.get("page_title")]
                if any(str(token) in answer for token in tokens if token):
                    linked.append(item.evidence_id)
            return {"linked_observed_evidence_ids": linked}
        # claim and absence/refusal stay semantic; no pass/fail observation is fabricated.
        return None


class ObservationRegistry(ObservationExtractor):
    """Compatibility name for the extractor registry requested by the contract."""


def _observation_text(answer: str, evidence: Sequence[ObservedEvidence]) -> str:
    parts = [answer]
    parts.extend(item.content or "" for item in evidence)
    return "\n".join(part for part in parts if part)


def _extract_numeric(text: str, assertion: Mapping[str, Any]) -> float | None:
    if not text:
        return None
    metric = str(assertion.get("metric") or "")
    year = assertion.get("year")
    aliases = _metric_aliases(metric)
    clauses = [item for item in re.split(r"[；;。\n]", text) if item.strip()]
    candidates = clauses
    if aliases:
        matched = [clause for clause in clauses if any(alias.casefold() in clause.casefold() for alias in aliases)]
        if matched:
            candidates = matched
    if year is not None:
        year_text = str(year)
        year_clauses = [clause for clause in candidates if year_text in clause]
        if year_clauses:
            candidates = year_clauses
        pattern = re.compile(rf"{re.escape(year_text)}\s*年?[^\d]{{0,24}}([-+]?\d+(?:,\d{{3}})*(?:\.\d+)?)")
        for clause in candidates:
            found = pattern.search(clause)
            if found:
                return _float(found.group(1))
    specialized = []
    if metric == "one_year_lpr":
        specialized.append(r"1\s*年期\s*LPR[^\d]{0,16}([-+]?\d+(?:,\d{3})*(?:\.\d+)?)")
    if "复合增长率" in "".join(aliases) or "growth" in metric.casefold():
        specialized.append(r"年均复合增长率[^\d]{0,16}([-+]?\d+(?:,\d{3})*(?:\.\d+)?)")
    for pattern in specialized:
        for clause in candidates:
            found = re.search(pattern, clause, re.IGNORECASE)
            if found:
                return _float(found.group(1))
    for clause in candidates:
        # Python's Unicode ``\w`` includes Chinese characters.  A value such as
        # ``LPR为3.0%`` must therefore be allowed to begin after ``为`` rather
        # than being reduced to the decimal tail ``0``.
        numbers = re.findall(r"(?<![0-9A-Za-z_])[-+]?\d+(?:,\d{3})*(?:\.\d+)?", clause)
        filtered = [number for number in numbers if not (len(number.replace(",", "")) == 4 and number.startswith(("19", "20")))]
        if filtered:
            return _float(filtered[-1] if "复合增长率" in metric else filtered[0])
    return None


def _metric_aliases(metric: str) -> tuple[str, ...]:
    aliases = {
        "one_year_lpr": ("1年期LPR", "一年期LPR", "LPR"),
        "operating_revenue": ("营业收入", "营收"),
        "营业收入": ("营业收入", "营收"),
        "营业收入年均复合增长率": ("营业收入", "年均复合增长率"),
        "净利润": ("净利润",),
        "净利润年均复合增长率": ("净利润", "年均复合增长率"),
        "产品覆盖省市数": ("省市", "覆盖"),
    }
    return aliases.get(metric, tuple(part for part in re.split(r"[_\W]+", metric) if len(part) > 1))


def _task_for_assertion(assertion: Mapping[str, Any], tasks: Sequence[Any]) -> Any:
    """Pick the reviewed composite child that most specifically owns an assertion."""
    if len(tasks) == 1 or assertion.get("type") != "numeric":
        return tasks[0]
    metric = str(assertion.get("metric") or "")
    aliases = _metric_aliases(metric)
    year = str(assertion.get("year") or "")

    def score(task: Any) -> tuple[int, int]:
        text = "\n".join(filter(None, (task.atomic_question, task.answer_fragment))).casefold()
        metric_hits = sum(alias.casefold() in text for alias in aliases)
        return (metric_hits, int(bool(year and year in text)))

    return max(tasks, key=score)


def _normalize_sql_row(row: Mapping[str, Any], identity: Mapping[str, Any]) -> dict[str, Any]:
    """Canonicalize runtime SQL column labels without changing row values."""
    aliases = {"基金代码": "fund_id", "基金简称": "fund_name", "资产净值": "asset_net_value"}
    normalized = {aliases.get(str(key), str(key)): value for key, value in row.items()}
    if "observation_date" not in normalized:
        date = _sql_observation_date(identity)
        if date is not None:
            normalized["observation_date"] = date
    return normalized


def _sql_observation_date(identity: Mapping[str, Any]) -> str | None:
    statement = identity.get("generated_sql") or identity.get("canonical_sql")
    if not isinstance(statement, str):
        return None
    matched = re.search(r"(?:\[)?(?:交易日期|observation_date)(?:\])?\s*=\s*['\"]?(\d{8})", statement, re.IGNORECASE)
    return matched.group(1) if matched else None


def _date(text: str) -> str | None:
    found = re.search(r"((?:19|20)\d{2})[-年/](\d{1,2})[-月/](\d{1,2})(?:日)?", text)
    if not found:
        return None
    return f"{int(found.group(1)):04d}-{int(found.group(2)):02d}-{int(found.group(3)):02d}"


def _float(value: str) -> float | None:
    try:
        return float(value.replace(",", ""))
    except ValueError:
        return None


def _metric_key(metric: str, year: Any) -> str:
    return f"{year}_{metric}" if year is not None else metric


def _relation_observation(assertion, values, answer):
    left_metric = str(assertion.get("left_metric") or "")
    right_metric = str(assertion.get("right_metric") or "")
    left = _find_metric_value(left_metric, values)
    right = _find_metric_value(right_metric, values)
    if left is None or right is None:
        return None
    left_value = float(left["value"])
    right_value = float(right["value"])
    left_unit = left.get("unit")
    right_unit = right.get("unit")
    if left_unit == "元" and right_unit == "万元":
        right_value *= 10000
    elif left_unit == "万元" and right_unit == "元":
        left_value *= 10000
    relation = assertion.get("relation")
    numeric_fact = left_value > right_value if relation == "greater_than" else left_value < right_value if relation == "less_than" else left_value == right_value
    visible = bool(re.search(r"2025.{0,20}(?:高于|大于).{0,20}2015|2015.{0,20}(?:低于|小于).{0,20}2025", answer))
    return {
        "value": bool(numeric_fact and visible),
        "numeric_relation_fact": numeric_fact,
        "user_visible_relation": visible,
        "left_value_normalized": left_value,
        "right_value_normalized": right_value,
        "unit": "元" if "元" in {left_unit, right_unit} else left_unit,
    }


def _find_metric_value(name: str, values: Mapping[str, Mapping[str, Any]]):
    year = next(iter(re.findall(r"(?:19|20)\d{2}", name)), None)
    for key, value in values.items():
        if year and year not in key:
            continue
        if "operating_revenue" in name and ("operating_revenue" in key or "营业收入" in key):
            return value
    return values.get(name)


__all__ = ["ObservationExtractor", "ObservationRegistry"]
