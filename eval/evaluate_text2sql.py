"""Run the real single-attempt benchmark: python -m eval.evaluate_text2sql."""

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from text2sql.schema_provider import DEFAULT_DB_PATH, get_schema_context
from text2sql.sql_generator import generate_sql
from tools.sql_executor import execute_sql
from tools.sql_validator import validate_sql

GOLD_PATH = ROOT / "eval/text2sql_gold.json"
RESULT_PATH = ROOT / "eval/text2sql_eval_result.json"
REPORT_PATH = ROOT / "TEXT2SQL_EVAL_REPORT.md"
COMPARISON_POLICY = (
    "比较完整结果，不比较 SQL 字符串；行顺序严格一致（本基准为排序查询或单行聚合）；"
    "列名和列顺序可不同，以整列值的一一匹配比较，缺列/多列失败；重复行保留。"
    "字符串严格比较（保留前导零），NULL 仅匹配 NULL；数值允许 int/float 等值，"
    "浮点容差 rel_tol=1e-12、abs_tol=1e-6；截断结果失败。"
)


def equal_value(actual, expected):
    if type(actual) in (int, float) and type(expected) in (int, float):
        return math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-6)
    return type(actual) is type(expected) and actual == expected


def compare_results(actual: dict, expected: list[dict]) -> tuple[bool, str | None]:
    """Match whole column vectors bijectively, ignoring aliases, never dropping cells."""
    if not actual["success"]:
        return False, "SQL execution failed"
    if actual["truncated"]:
        return False, "Execution result was truncated"
    rows = actual["rows"]
    if len(rows) != len(expected):
        return False, f"Row count differs: actual={len(rows)}, gold={len(expected)}"
    if not expected:
        return True, None
    actual_columns = actual["columns"]
    expected_columns = list(expected[0])
    if len(set(actual_columns)) != len(actual_columns):
        return False, "Duplicate output column names lose values in executor row dictionaries"
    if len(actual_columns) != len(expected_columns):
        return False, (f"Column count differs: actual={actual_columns}, gold={expected_columns}")
    candidates = {
        key: [name for name in actual_columns
              if all(name in row and equal_value(row[name], gold[key])
                     for row, gold in zip(rows, expected))]
        for key in expected_columns
    }
    # Bipartite matching handles repeated/equal columns without a greedy mismatch.
    owners = {}

    def assign(key, seen):
        for name in candidates[key]:
            if name in seen:
                continue
            seen.add(name)
            if name not in owners or assign(owners[name], seen):
                owners[name] = key
                return True
        return False

    for key in expected_columns:
        if not assign(key, set()):
            return False, f"No equivalent output column for gold [{key}] (values/types/order differ)"
    return True, None


def evaluate_case(case):
    started = perf_counter()
    # The generator receives ONLY question text. Gold SQL is never used here.
    generation = generate_sql(case["question"])
    sql = generation["sql"]
    validation = validate_sql(sql)
    record = dict(
        id=case["id"], question=case["question"], generated_sql=sql,
        raw_output=generation["raw_output"],
        generation_success=generation["generation_success"],
        validation_pass=validation["valid"], execution_success=False, result_match=False,
        error_type=generation["error_type"], error=generation["error"],
        generation_latency_ms=generation["latency_ms"], execution_latency_ms=None,
        actual_result=None, gold_result=case["gold_result"], latency_ms=0.0,
    )
    if generation["generation_success"] and not validation["valid"]:
        record.update(error_type="validation_error", error=validation["error"])
    elif generation["success"] and validation["valid"]:
        execution = execute_sql(sql, str(DEFAULT_DB_PATH))
        match, reason = compare_results(execution, case["gold_result"])
        record.update(execution_success=execution["success"], result_match=match,
                      actual_result=execution, execution_latency_ms=execution["latency_ms"],
                      error_type=(None if match else execution["error_type"] or "result_mismatch"),
                      error=None if match else execution["error"] or reason)
    record["latency_ms"] = round((perf_counter() - started) * 1000, 4)
    return record


def write_report(output):
    metrics = output["metrics"]
    lines = ["# Text-to-SQL Phase 1 Evaluation", "",
             f"- 时间（UTC）：{output['evaluated_at']}",
             f"- 模型：{output['model']}；temperature=0.1；每题单次请求，无重试/反思/修复。",
             f"- Schema Context：{output['schema_context_chars']} 字符，"
             f"{output['schema_context_utf8_bytes']} UTF-8 字节。",
             f"- Gold SHA-256：`{output['gold_sha256']}`；评估前后校验一致。",
             "- Generator 仅接收问题与 Schema，不接收题号、Gold SQL 或 Gold Result。",
             "", "## 指标", "",
             "所有比率分母均为完整题数。Generation Success 表示模型返回非空文本；"
             "Validation Pass 表示现有词法策略通过；SQL Execution Accuracy 表示执行成功率，"
             "不代表语义正确；Result Accuracy 表示最终结果匹配率。", "",
             "| 指标 | 成功/总数 | 比率 |", "|---|---:|---:|"]
    for name, metric in metrics.items():
        lines.append(f"| {name} | {metric['count']}/{output['total']} | {metric['rate']:.1%} |")
    lines += ["", "## 结果比较口径", "", COMPARISON_POLICY,
              "", "## 逐题结果", "",
              "| ID | Generation | Validation | Execution | Result Match | 延迟 ms |",
              "|---|---|---|---|---|---:|"]
    for row in output["cases"]:
        flags = [str(row[key]) for key in ("generation_success", "validation_pass",
                                          "execution_success", "result_match")]
        lines.append(f"| {row['id']} | " + " | ".join(flags) + f" | {row['latency_ms']:.1f} |")
    lines += ["", "## 每题 SQL 与失败原因", ""]
    for row in output["cases"]:
        lines += [f"### {row['id']}", "", row["question"], "", "```sql",
                  row["generated_sql"] or "(无 SQL)", "```", "",
                  "结果匹配。" if row["result_match"] else f"失败类型：{row['error_type']}；原因：{row['error']}", ""]
    lines += ["## 限制", "", "现有 Validator 是词法策略检查，SQL 语法和字段错误在执行阶段发现；"
              "Schema 的 JOIN/业务规则由提示约束，尚未增加 AST 级语义校验。"
              "本报告只反映当前模型在这 10 题的一次生成表现。"
              "严格完整结果比较会将题意未明确列出的上下文字段缺失也记为失败；"
              "可在 JSON 中核查实际结果，但本轮不会为提高成绩调整口径。", "",
              "完整原始输出、实际结果和 Gold Result：`eval/text2sql_eval_result.json`。"]
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    original = GOLD_PATH.read_bytes()
    cases = json.loads(original)
    if len(cases) != 10:
        raise ValueError("Phase 1 requires exactly 10 benchmark questions")
    context = get_schema_context()
    output = dict(evaluated_at=datetime.now(timezone.utc).isoformat(), model=config.LLM_MODEL,
                  total=len(cases), schema_context_chars=len(context),
                  schema_context_utf8_bytes=len(context.encode("utf-8")),
                  gold_sha256=hashlib.sha256(original).hexdigest(),
                  comparison_policy=COMPARISON_POLICY, cases=[], metrics={})
    for case in cases:
        record = evaluate_case(case)
        output["cases"].append(record)
        print(f"{record['id']}: generation={record['generation_success']} "
              f"validation={record['validation_pass']} execution={record['execution_success']} "
              f"match={record['result_match']}", flush=True)
        RESULT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    if GOLD_PATH.read_bytes() != original:
        raise RuntimeError("Gold file changed during evaluation")
    for name, key in [("Generation Success Rate", "generation_success"),
                      ("SQL Validation Pass Rate", "validation_pass"),
                      ("SQL Execution Accuracy", "execution_success"),
                      ("Result Accuracy", "result_match")]:
        count = sum(row[key] for row in output["cases"])
        output["metrics"][name] = {"count": count, "rate": count / len(cases)}
    RESULT_PATH.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(output)
    print(json.dumps(output["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
