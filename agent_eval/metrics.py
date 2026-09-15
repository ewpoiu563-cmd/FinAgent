"""Evaluator headline metrics and Wilson confidence intervals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

from .models import CaseEvaluation


@dataclass(frozen=True, slots=True)
class RatioMetric:
    numerator: float
    denominator: float
    value: float | None
    wilson_low: float | None
    wilson_high: float | None
    invalid_count: int = 0


def ratio(numerator: float, denominator: float, *, invalid_count: int = 0, wilson: bool = True) -> RatioMetric:
    if denominator == 0:
        return RatioMetric(numerator, denominator, None, None, None, invalid_count)
    value = numerator / denominator
    # Wilson is defined for Bernoulli counts. Weighted AC still reports the
    # same bounded effective-count interval over assertion weight.
    z = 1.959963984540054
    center = (value + z*z/(2*denominator)) / (1 + z*z/denominator)
    margin = z * math.sqrt(value*(1-value)/denominator + z*z/(4*denominator*denominator)) / (1 + z*z/denominator)
    if not wilson:
        return RatioMetric(numerator, denominator, value, None, None, invalid_count)
    return RatioMetric(numerator, denominator, value, max(0.0, center-margin), min(1.0, center+margin), invalid_count)


def aggregate_evaluations(results: Iterable[CaseEvaluation]) -> Mapping[str, RatioMetric]:
    cases = tuple(results)
    requirements = tuple(item for case in cases for item in case.requirement_results if item.required)
    headline = tuple(item for item in requirements if item.headline_eligible)
    assertions = tuple(item for requirement in headline for item in requirement.assertion_results)
    invalid_cases = sum(not case.valid_artifact for case in cases)
    invalid_requirements = sum(bool(item.invalid_reasons) for item in requirements)
    eligible_cases = tuple(case for case in cases if case.headline_eligible)
    return {
        "tsr": ratio(sum(case.tsr_contribution for case in eligible_cases), len(eligible_cases), invalid_count=invalid_cases),
        "grpr": ratio(sum(item.planning_recall for item in requirements), len(requirements), invalid_count=invalid_requirements),
        "tna": ratio(sum(item.tool_necessity_correct for item in headline), len(headline), invalid_count=invalid_requirements),
        "ssa": ratio(sum(item.planned_source_correct for item in headline), len(headline), invalid_count=invalid_requirements),
        "strict_answer_correctness": ratio(sum(item.answer_correct_strict for item in headline), len(headline), invalid_count=invalid_requirements),
        "weighted_answer_correctness": ratio(sum(assertion.weight for assertion in assertions if assertion.passed), sum(assertion.weight for assertion in assertions), invalid_count=invalid_requirements, wilson=False),
        "gold_requirement_coverage": ratio(sum(item.weight for item in headline if item.requirement_fully_satisfied), sum(item.weight for item in headline), invalid_count=invalid_requirements, wilson=False),
        "actual_source_compliance": ratio(sum(item.actual_source_compliant for item in headline), len(headline), invalid_count=invalid_requirements),
        "fallback_compliance": ratio(sum(item.fallback_compliant for item in headline), len(headline), invalid_count=invalid_requirements),
    }
