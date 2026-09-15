"""Contracts for user-scoped Web source constraints."""

import asyncio

from orchestration.web_fallback import ControlledWebFallbackExecutor
from orchestration.web_source_policy import WebSourceMode, parse_web_source_policy


def test_explicit_csrc_constraint_is_allowlist_not_global_default():
    restricted = parse_web_source_policy("只使用证监会官网，不要媒体转载，说明注册制规则。")
    ordinary = parse_web_source_policy("法国的首都是什么？")

    assert restricted.mode is WebSourceMode.ALLOWLIST
    assert restricted.permits_url("https://www.csrc.gov.cn/csrc/c100028/common_list.shtml")
    assert not restricted.permits_url("https://example-news.com/story")
    assert ordinary.mode is WebSourceMode.OPEN
    assert ordinary.permits_url("https://www.service-public.fr/particuliers")


def test_literal_domain_constraint_and_local_only_constraint():
    domain_limited = parse_web_source_policy("只用 https://www.who.int 的官方页面回答。")
    local_only = parse_web_source_policy("只查本地数据库，不要联网。")

    assert domain_limited.mode is WebSourceMode.ALLOWLIST
    assert domain_limited.permits_url("https://www.who.int/news-room")
    assert not domain_limited.permits_url("https://news.example.org/repost")
    assert local_only.mode is WebSourceMode.PROHIBITED
    assert not local_only.web_permitted


def test_named_official_source_is_a_request_allowlist_without_only_wording():
    policy = parse_web_source_policy("请查中国人民银行官方公告中的 LPR。")
    assert policy.mode is WebSourceMode.ALLOWLIST
    assert policy.permits_url("https://www.pbc.gov.cn/goutongjiaoliu/113456/113469/index.html")
    assert not policy.permits_url("https://example-news.org/lpr-repost")


def test_executor_filters_before_evidence_evaluation_when_domain_is_limited():
    async def search(*_args, **_kwargs):
        return [{"url": "https://news.example.org/repost", "title": "repost", "snippet": "x"}]

    result = asyncio.run(
        ControlledWebFallbackExecutor(search_call=search, max_fetches=0).execute(
            "只使用证监会官网回答。",
            web_source_policy=parse_web_source_policy("只使用证监会官网回答。"),
        )
    )

    assert result.status == "insufficient"
    assert "来源限制" in result.missing_information[0]
