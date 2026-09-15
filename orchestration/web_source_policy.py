"""Task-scoped Web source constraints and transparent authority defaults.

An explicit source restriction belongs to the user's *whole request*.  It is
not a global preference for one financial regulator and it must survive task
decomposition.  In the normal open mode this module does not discard ordinary
sites; executors may rank them with domain-specific authority rules instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from urllib.parse import urlparse


class WebSourceMode(str, Enum):
    OPEN = "open"
    ALLOWLIST = "allowlist"
    PROHIBITED = "prohibited"


@dataclass(frozen=True)
class WebSourcePolicy:
    mode: WebSourceMode = WebSourceMode.OPEN
    allowed_domains: tuple[str, ...] = ()
    require_primary_source: bool = False
    authority_profile: str = "domain_aware"
    reason: str = "No explicit Web source restriction"

    @property
    def web_permitted(self) -> bool:
        return self.mode is not WebSourceMode.PROHIBITED

    def permits_url(self, url: str) -> bool:
        if not self.web_permitted:
            return False
        if self.mode is WebSourceMode.OPEN:
            return True
        host = (urlparse(url).hostname or "").lower().rstrip(".")
        return any(host == domain or host.endswith(f".{domain}") for domain in self.allowed_domains)

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "allowed_domains": list(self.allowed_domains),
            "require_primary_source": self.require_primary_source,
            "authority_profile": self.authority_profile,
            "reason": self.reason,
        }


# This map is only used after the user names an institution.  It is deliberately
# not a default allowlist for unrelated Web questions.
_OFFICIAL_DOMAIN_ALIASES = {
    "证监会": "csrc.gov.cn",
    "中国证监会": "csrc.gov.cn",
    "人民银行": "pbc.gov.cn",
    "中国人民银行": "pbc.gov.cn",
    "上交所": "sse.com.cn",
    "上海证券交易所": "sse.com.cn",
    "深交所": "szse.cn",
    "深圳证券交易所": "szse.cn",
}


def parse_web_source_policy(question: str) -> WebSourcePolicy:
    """Parse only explicit user constraints; otherwise leave the Web open."""
    text = question.lower()
    local_only = bool(re.search(r"(?:只|仅|仅限|只能|限于).{0,16}(?:本地数据库|数据库|sql)", text))
    # A per-clause local-only constraint must not suppress an explicitly
    # requested external-document/Web clause, regardless of clause order.
    # Bare "年报" is intentionally excluded: it is also a common SQL report
    # type and would otherwise turn local-only analytical questions into Web.
    mixed_external_request = (
        local_only
        and any(marker in text for marker in (
            "招股书", "公告", "官网", "证监会", "人民银行", "上交所", "深交所", "网页", "web",
        ))
        and any(connector in text for connector in ("再", "同时", "并且", "并", "以及"))
    )
    if (local_only and not mixed_external_request) or re.search(
        r"(?:不要联网|不联网|不查网页|禁止网页|禁止web)", text
    ):
        return WebSourcePolicy(
            mode=WebSourceMode.PROHIBITED,
            authority_profile="local_only",
            reason="User explicitly prohibited Web access or limited the request to local SQL data",
        )

    # Keep this deliberately simple and Unicode-safe.  A literal domain plus
    # a limiting word is enough; the more permissive pattern covers requests
    # such as "只使用证监会官网" that rely on an institution alias.
    named_official_source = any(alias in question for alias in _OFFICIAL_DOMAIN_ALIASES) and any(
        marker in question for marker in ("官网", "官方", "查询", "查证", "根据")
    )
    explicitly_limited = (
        bool(re.search(r"(?:只|仅|仅限|只能|限于|不要).{0,20}(?:官网|官方|媒体|转载|来源|网站)", text))
        or (
            any(marker in question for marker in ("只", "仅", "只能", "限于", "不要"))
            and any(marker in question for marker in ("官网", "官方", "媒体", "转载", "来源", "网站"))
        )
        or named_official_source
    )
    domains = set(re.findall(r"(?<![\w.-])(?:https?://)?([a-z0-9-]+(?:\.[a-z0-9-]+)+)", text))
    if explicitly_limited:
        for alias, domain in _OFFICIAL_DOMAIN_ALIASES.items():
            if alias in question:
                domains.add(domain)
        if domains:
            return WebSourcePolicy(
                mode=WebSourceMode.ALLOWLIST,
                allowed_domains=tuple(sorted(domains)),
                require_primary_source=True,
                authority_profile="explicit_primary_source",
                reason="User explicitly limited acceptable Web sources",
            )
    return WebSourcePolicy()
