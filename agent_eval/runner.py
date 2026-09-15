"""Repeatable evaluator-side runtime runner and artifact persistence."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from orchestration.trace import JsonlTraceSink, TraceRecorder, runtime_artifact_observer

from .alignment import DeterministicRequirementAligner, RequirementAligner, ReviewedMappingAligner
from .artifact_builder import EvaluationRunArtifactBuilder
from .assertions import AssertionValidatorRegistry, SemanticJudge
from .gold import load_gold
from .models import CaseEvaluation, EvaluationRunArtifact
from .observations import ObservationExtractor
from .runtime_capture import RuntimeCapture
from .scoring import AgentEvaluator
from .semantic_judge import SemanticJudgeAdapter


@dataclass(frozen=True, slots=True)
class EvaluationRunnerResult:
    run_artifact: EvaluationRunArtifact
    evaluation_result: CaseEvaluation | None
    run_directory: Path


class EvaluationRunner:
    def __init__(
        self,
        *,
        entrypoint: Any,
        trace_path: str | Path,
        run_summary_path: str | Path,
        output_dir: str | Path = "outputs/eval",
        builder: EvaluationRunArtifactBuilder | None = None,
        aligner: RequirementAligner | None = None,
        semantic_judge: SemanticJudge | None = None,
    ) -> None:
        self.entrypoint = entrypoint
        self.trace_path = Path(trace_path)
        self.run_summary_path = Path(run_summary_path)
        self.output_dir = Path(output_dir)
        self.builder = builder or EvaluationRunArtifactBuilder()
        self.aligner = aligner or DeterministicRequirementAligner()
        self.semantic_judge = semantic_judge

    async def run_case(
        self,
        gold_case: Mapping[str, Any],
        *,
        system_snapshot: Mapping[str, Any],
        score: bool = True,
        session_id: str | None = None,
    ) -> EvaluationRunnerResult:
        capture = RuntimeCapture()
        with runtime_artifact_observer(capture):
            final_answer = await self.entrypoint.run(gold_case["question"], session_id=session_id)
        artifact = self.builder.build(
            case_id=gold_case["case_id"],
            reference_date=gold_case["reference_date"],
            final_answer=final_answer,
            capture=capture,
            trace_path=self.trace_path,
            run_summary_path=self.run_summary_path,
            system_snapshot=system_snapshot,
        )
        artifact = ObservationExtractor(aligner=self.aligner).extract(gold_case, artifact)
        run_directory = self.output_dir / "runs" / artifact.run_id
        run_directory.mkdir(parents=True, exist_ok=True)
        evaluation = None
        if score:
            if isinstance(self.semantic_judge, SemanticJudgeAdapter) and self.semantic_judge.log_path is None:
                self.semantic_judge.log_path = run_directory / "judge_log.jsonl"
            evaluator = AgentEvaluator(
                aligner=self.aligner,
                validators=AssertionValidatorRegistry(self.semantic_judge),
            )
            evaluation = evaluator.evaluate(gold_case, artifact)
        _persist_json(run_directory / "run_artifact.json", asdict(artifact))
        if evaluation is not None:
            _persist_json(run_directory / "evaluation_result.json", asdict(evaluation))
        return EvaluationRunnerResult(artifact, evaluation, run_directory)


def _persist_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _system_snapshot(gold_path: Path, reference_date: str) -> Mapping[str, Any]:
    return {
        "phase": "5.3b",
        "reference_date": reference_date,
        "python": platform.python_version(),
        "gold_file": str(gold_path),
        "gold_sha256": hashlib.sha256(gold_path.read_bytes()).hexdigest(),
        "trace_schema_version": "1.0",
        "run_summary_schema_version": "1.1",
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one FinAgent evaluation case")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--gold-file", default="eval/gold/agent_eval_gold_v0.1.json")
    parser.add_argument("--output-dir", default="outputs/eval")
    parser.add_argument("--reference-date")
    parser.add_argument("--trace-dir", default="outputs/traces")
    parser.add_argument("--mapping-file")
    parser.add_argument("--dry-run", "--no-score", action="store_true", dest="no_score", help="run and persist observations without scoring")
    parser.add_argument("--semantic-judge", choices=("on", "off"), default="off")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    gold_path = Path(args.gold_file)
    gold = load_gold(gold_path)
    case = next((item for item in gold["cases"] if item["case_id"] == args.case_id), None)
    if case is None:
        raise SystemExit(f"unknown case_id: {args.case_id}")
    if args.reference_date and args.reference_date != case["reference_date"]:
        raise SystemExit("--reference-date must equal the immutable Gold reference_date")
    reference = date.fromisoformat(args.reference_date or case["reference_date"])
    from orchestration.entrypoint import build_default_entrypoint
    from react_agent import run_react_agent

    entrypoint = build_default_entrypoint(run_react_agent, reference_date=reference)
    trace_dir = Path(args.trace_dir)
    trace_path = trace_dir / "trace.jsonl"
    summary_path = trace_dir / "run_summaries.jsonl"
    entrypoint.trace_recorder = TraceRecorder(JsonlTraceSink(trace_path, summary_path=summary_path))
    aligner = ReviewedMappingAligner.from_json(args.mapping_file) if args.mapping_file else DeterministicRequirementAligner()
    judge = None
    if args.semantic_judge == "on":
        from config import call_llm

        judge = SemanticJudgeAdapter(call_llm)
    runner = EvaluationRunner(
        entrypoint=entrypoint,
        trace_path=trace_path,
        run_summary_path=summary_path,
        output_dir=args.output_dir,
        aligner=aligner,
        semantic_judge=judge,
    )
    result = asyncio.run(
        runner.run_case(
            case,
            system_snapshot=_system_snapshot(gold_path, reference.isoformat()),
            score=not args.no_score,
        )
    )
    print(json.dumps({"run_id": result.run_artifact.run_id, "run_directory": str(result.run_directory), "scored": result.evaluation_result is not None}, ensure_ascii=False))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = ["EvaluationRunner", "EvaluationRunnerResult", "main"]
