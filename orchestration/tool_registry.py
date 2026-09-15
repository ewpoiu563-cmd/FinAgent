"""Single source of truth for FinAgent tool capabilities and contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .models import SourceType


@dataclass(frozen=True)
class ToolSpec:
    name: str
    source_type: SourceType
    description: str
    capabilities: tuple[str, ...]
    exclusions: tuple[str, ...]
    input_contract: Mapping[str, Any]
    output_contract: Mapping[str, Any]
    routing_hints: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.description.strip():
            raise ValueError("tool name and description must be non-empty")
        if not self.capabilities or not self.exclusions:
            raise ValueError(f"tool {self.name} must declare capabilities and exclusions")


class ToolRegistry:
    def __init__(self, specs: Iterable[ToolSpec] = ()) -> None:
        self._specs: dict[str, ToolSpec] = {}
        for spec in specs:
            self.register(spec)

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._specs:
            raise ValueError(f"tool already registered: {spec.name}")
        self._specs[spec.name] = spec

    def get(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError as error:
            raise KeyError(f"unknown tool: {name}") from error

    def all(self) -> tuple[ToolSpec, ...]:
        return tuple(self._specs.values())

    def by_source(self, source_type: SourceType) -> tuple[ToolSpec, ...]:
        return tuple(spec for spec in self._specs.values() if spec.source_type == source_type)

    def routing_hints(self, source_type: SourceType) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                hint
                for spec in self.by_source(source_type)
                for hint in spec.routing_hints
            )
        )

    def render_capabilities(self) -> str:
        blocks = []
        for spec in self.all():
            capabilities = "；".join(spec.capabilities)
            exclusions = "；".join(spec.exclusions)
            blocks.append(
                f"{spec.name} [{spec.source_type.value}]\n"
                f"用途：{spec.description}\n能力：{capabilities}\n不适用：{exclusions}"
            )
        return "\n\n".join(blocks)


def build_default_tool_registry() -> ToolRegistry:
    natural_language_input = {
        "type": "object",
        "required": ["question"],
        "properties": {"question": {"type": "string", "minLength": 1}},
    }
    return ToolRegistry(
        [
            ToolSpec(
                name="query_financial_db",
                source_type=SourceType.SQL,
                description="查询本地 finance.db 中已经验证的结构化基金数据。",
                capabilities=(
                    "基金基本信息",
                    "基金股票持仓",
                    "基金日行情",
                    "基金规模",
                    "Top-K 与排名",
                    "聚合统计",
                ),
                exclusions=("招股书正文", "新闻", "数据库 schema 外的信息", "实时行情"),
                input_contract=natural_language_input,
                output_contract={
                    "type": "object",
                    "fields": (
                        "success", "generated_sql", "columns", "rows", "row_count",
                        "truncated", "error_type", "error", "latency_ms",
                    ),
                },
                routing_hints=(
                    "基金基本信息", "基金股票持仓", "基金持仓", "持有", "基金日行情",
                    "基金规模", "资产净值", "top-k", "topk", "排名", "最多", "最少",
                    "聚合", "统计", "平均", "合计", "数量",
                ),
            ),
            ToolSpec(
                name="retrieve_document",
                source_type=SourceType.RAG,
                description="从本地金融文档索引中检索可引用证据。",
                capabilities=("招股说明书", "年报与财报", "公司业务", "风险", "客户供应商", "历史披露"),
                exclusions=("最新新闻", "实时行情", "基金数据库查询"),
                input_contract={
                    **natural_language_input,
                    "properties": {
                        **natural_language_input["properties"],
                        "allowed_doc_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "内部执行边界使用的物理 index_doc_id；Planner 不得生成该字段。",
                        },
                    },
                },
                output_contract={
                    "type": "object",
                    "fields": (
                        "success", "query", "evidence", "retrieval_metadata", "degraded",
                        "fallback", "error_type", "error", "latency_ms",
                    ),
                },
                routing_hints=(
                    "招股说明书", "招股书", "年报", "年度报告", "财报", "财务报告",
                    "pdf", "本地文档", "报告中", "披露文件中",
                ),
            ),
            ToolSpec(
                name="search",
                source_type=SourceType.WEB,
                description="搜索最新或当前公开 Web 信息并返回标题、URL 与摘要。",
                capabilities=("最新新闻", "当前公开信息", "近期公告", "Web 补充信息"),
                exclusions=("本地文档限定问题", "finance.db 精确查询"),
                input_contract={
                    "type": "object",
                    "required": ["query"],
                    "properties": {"query": {"type": "string", "minLength": 1}},
                },
                output_contract={"type": "array", "item_fields": ("title", "url", "snippet", "source_type")},
                routing_hints=("最新", "最近", "近期", "今天", "今日", "新闻", "当前动态", "最新公告", "实时"),
            ),
            ToolSpec(
                name="fetch",
                source_type=SourceType.WEB,
                description="抓取 Search 已发现 URL 的正文，用于补充 Web 证据。",
                capabilities=("网页正文", "公开页面详情", "搜索摘要补充"),
                exclusions=("未经过 Search 发现的任意 URL", "本地 RAG 文档检索", "金融数据库查询"),
                input_contract={
                    "type": "object",
                    "required": ["url"],
                    "properties": {"url": {"type": "string", "format": "uri"}},
                },
                output_contract={"type": "string", "description": "bounded page text"},
            ),
        ]
    )
