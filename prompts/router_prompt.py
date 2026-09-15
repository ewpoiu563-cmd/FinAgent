"""Bounded prompt for ambiguous source routing only."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestration.models import EntityResolution
    from orchestration.tool_registry import ToolRegistry


def build_router_prompt(
    question: str,
    entity: EntityResolution,
    registry: ToolRegistry,
    *,
    allowed_sources=None,
    rejected_sources=(),
    rejection_reason: str | None = None,
) -> str:
    recovery = ""
    if allowed_sources is not None:
        recovery = (
            "\n这是一次来源纠错重规划："
            f"只允许选择 {[source.value for source in allowed_sources]}；"
            f"不得再选择 {[source.value for source in rejected_sources]}；"
            f"上次决策被拒绝的原因是 {rejection_reason or 'source capability mismatch'}。\n"
        )
    return (
        "你是 FinAgent 的轻量信息源路由器。上游已判定该任务存在外部信息缺口；"
        "你只判断完成问题需要哪些来源，并选择填补该缺口所需的最小充分来源集合，"
        "不回答问题，也不生成工具调用。"
        "不要为了保险追加 Web，不要重新判断 no_tool。只允许 sql、rag、web。\n"
        "sql 仅限 registry 中声明的本地结构化基金数据库能力；"
        "rag 仅限本地金融文档；web 用于最新或当前公开信息。\n"
        "只返回一个 JSON 对象，字段必须为：primary_source、required_sources、"
        "web_fallback_allowed、reason。required_sources 必须是非空且无重复的字符串数组，"
        "primary_source 必须包含在其中。只有确有第二来源无法替代的独立证据需求时才选择多个来源；"
        "optional source 不得放入 required_sources。\n\n"
        f"工具能力：\n{registry.render_capabilities()}\n\n"
        f"已解析业务实体：{json.dumps(entity.to_dict(), ensure_ascii=False)}\n"
        f"用户问题：{question}"
        f"{recovery}"
    )
