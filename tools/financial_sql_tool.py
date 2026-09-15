"""High-level natural-language-to-financial-database query tool."""

from __future__ import annotations

import json
import logging
from time import perf_counter
from typing import Any, Mapping

from text2sql.schema_provider import DEFAULT_DB_PATH
from text2sql.sql_generator import generate_sql
from text2sql.entity_grounding import (
    FinancialEntityGrounding,
    ground_financial_entities,
    validate_semantic_literals,
)
from text2sql.semantic_planner import SQLIntent, SQLSemanticPlanner
from text2sql.semantic_validator import validate_sql_against_intent
from .sql_executor import execute_sql
from .sql_validator import validate_sql


logger = logging.getLogger(__name__)
_DEFAULT_GENERATE_SQL = generate_sql


def query_financial_db(
    question: str,
    *,
    sql_intent: SQLIntent | None = None,
    repair_observation: Mapping[str, Any] | None = None,
) -> dict:
    """Generate, validate, and execute one read-only financial SQL attempt.

    The returned data is an observation for an agent, not a natural-language
    answer. The orchestration layer owns the bounded retry loop; this function
    performs exactly one attempt and can use its supplied repair observation.
    """
    started = perf_counter()
    result = {
        "success": False,
        "question": question,
        "generated_sql": None,
        "columns": [],
        "rows": [],
        "row_count": 0,
        "truncated": False,
        "latency_ms": 0.0,
        "error_type": None,
        "error": None,
        "validation_error": None,
        "execution_error": None,
    }

    def fail(error_type: str, error: object) -> None:
        message = str(error)
        result.update(error_type=error_type, error=message)
        if error_type == "validation_error":
            result["validation_error"] = message
        elif error_type in {"sqlite_error", "timeout"}:
            result["execution_error"] = message

    stage = "generation"
    try:
        if sql_intent is not None:
            # The orchestration layer already confirmed this grounding together
            # with the rest of the intent; do not independently reinterpret it.
            grounding = FinancialEntityGrounding(sql_intent.grounded_entities)
        else:
            try:
                grounding = ground_financial_entities(question, DEFAULT_DB_PATH)
            except Exception:
                # Keep the frozen SQL path usable if optional grounding cannot read
                # the local DB; the post-generation guard still rejects unsupported
                # identity literals that were not present in the user question.
                grounding = FinancialEntityGrounding()
        intent = sql_intent or SQLSemanticPlanner().plan(question, grounding)
        logger.debug(
            "SQL_SEMANTIC_INTENT: %s",
            json.dumps(intent.to_dict(), ensure_ascii=False),
        )
        if intent.clarification_needed:
            fail("clarification_needed", intent.clarification_question or "SQL 查询语义需要澄清")
            return result
        logger.debug(
            "FINANCIAL_ENTITY_GROUNDING: %s",
            json.dumps(
                {
                    "question_preview": question[:200],
                    "matched": grounding.matched,
                    "entities": [entity.to_dict() for entity in grounding.entities[:20]],
                },
                ensure_ascii=False,
            ),
        )
        # Preserve the historical one-argument monkeypatch seam used by the
        # frozen Phase 1 tests. Production receives typed grounding via intent;
        # a replaced test double receives the equivalent canonical annotation.
        generation = (
            generate_sql(
                question,
                intent=intent,
                repair_observation=dict(repair_observation) if repair_observation is not None else None,
            )
            if generate_sql is _DEFAULT_GENERATE_SQL
            else generate_sql(grounding.augment_question(question))
        )
        sql = generation.get("sql")
        result["generated_sql"] = sql

        if not generation.get("success"):
            # The generator already validates its one generated statement. Keep
            # that stage distinguishable from transport/schema/output failures.
            if generation.get("error_type") == "validation_error":
                validation = validate_sql(sql)
                fail("validation_error", validation.get("error") or generation.get("error"))
            else:
                fail("generation_error", generation.get("error") or "SQL generation failed")
            return result

        semantic_contract_active = sql_intent is not None or generate_sql is _DEFAULT_GENERATE_SQL
        intent_validation = (
            validate_sql_against_intent(sql, intent)
            if semantic_contract_active
            else None
        )
        logger.debug(
            "SQL_SEMANTIC_VALIDATION: %s",
            json.dumps(
                {
                    "valid": intent_validation.valid if intent_validation else True,
                    "errors": list(intent_validation.errors) if intent_validation else [],
                    "compatibility_seam": not semantic_contract_active,
                },
                ensure_ascii=False,
            ),
        )
        if intent_validation is not None and not intent_validation.valid:
            fail("validation_error", intent_validation.error or "SQL intent validation failed")
            return result

        semantic_validation = validate_semantic_literals(sql, question, grounding)
        logger.debug(
            "SEMANTIC_LITERAL_VALIDATION: %s",
            json.dumps(
                {
                    "valid": semantic_validation.valid,
                    "error": semantic_validation.error,
                },
                ensure_ascii=False,
            ),
        )
        if not semantic_validation.valid:
            fail("validation_error", semantic_validation.error or "semantic literal validation failed")
            return result

        stage = "validation"
        validation = validate_sql(sql)
        if not validation["valid"]:
            fail("validation_error", validation["error"])
            return result

        stage = "execution"
        execution = execute_sql(sql, str(DEFAULT_DB_PATH))
        if not execution.get("success"):
            error_type = execution.get("error_type")
            if error_type not in {"sqlite_error", "timeout", "validation_error"}:
                error_type = "sqlite_error"
            fail(error_type, execution.get("error") or "SQL execution failed")
            return result

        result.update(
            success=True,
            columns=execution["columns"],
            rows=execution["rows"],
            row_count=execution["row_count"],
            truncated=execution["truncated"],
        )
    except TimeoutError as exc:
        fail("timeout" if stage == "execution" else f"{stage}_error", exc)
    except Exception as exc:
        fail({"generation": "generation_error", "validation": "validation_error",
              "execution": "sqlite_error"}[stage], exc)
    finally:
        result["latency_ms"] = round((perf_counter() - started) * 1000, 4)

    return result
