"""Small evidence-only prompt for two-source FinAgent synthesis."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any


_SOURCE_TITLES = {
    "rag": "RAG Evidence (本地文档)",
    "sql": "SQL Evidence (本地结构化数据库)",
    "web": "Web Evidence (公开网页)",
}


def build_multi_source_synthesis_prompt(
    question: str,
    evidence_by_source: Mapping[str, Sequence[Mapping[str, Any]]],
) -> str:
    blocks = [
        "你是 FinAgent 的多源证据综合器。只允许使用下列已验证 evidence，不得使用模型先验、Thought 或外部事实。",
        "每个事实 claim 都必须列出真实存在的 supported_evidence_ids。不得把 Web 证据写成本地文档披露，也不得把本地文档证据描述成最新公开信息。",
        "最终 answer 必须明确使用来源措辞，例如‘本地文档显示’、‘数据库结果显示’、‘Web 公开信息显示’。",
        "只返回 JSON：status 必须为 success；answer 为非空字符串；claims 为 claim/supported_evidence_ids 对象数组；source_summary 为逐来源摘要对象。",
        "",
        "Original Question",
        question.strip(),
    ]
    for source, items in evidence_by_source.items():
        blocks.extend(
            [
                "",
                _SOURCE_TITLES[source],
                json.dumps(list(items), ensure_ascii=False, default=str),
            ]
        )
    return "\n".join(blocks)
