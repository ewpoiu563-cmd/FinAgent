"""Bounded navigation of official paginated listing pages.

Chinese government and exchange sites publish notices as a paged list whose
entries only exist on later numbered pages, and those detail pages are often
absent from search-engine results for a domain-restricted query. When the
question supplies a concrete date, one bounded traversal of the listing's own
numbered pages recovers the official detail page instead of widening the
source allowlist.

The module is deliberately pure except for ``fetch_raw_html`` so the parsing
rules can be unit tested against saved markup.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from html import unescape
from typing import Iterable, Sequence
from urllib.parse import urljoin, urlsplit


_PAGE_HREF = re.compile(r"([A-Za-z0-9_\-]+)-(\d+)\.html", re.IGNORECASE)
_ANCHOR = re.compile(r"<a\b[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>", re.IGNORECASE | re.DOTALL)
_TITLE_ATTR = re.compile(r"title=[\"']([^\"']+)[\"']", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class ListingEntry:
    """One notice entry discovered on an official listing page."""

    url: str
    title: str

    def as_search_result(self) -> dict[str, str]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.title,
            "source_type": "organic",
        }


def date_tokens(text: str) -> tuple[str, ...]:
    """Return the written forms a listing may use for the question's dates."""

    tokens: list[str] = []
    for year, month, day in re.findall(r"((?:19|20)\d{2})年(\d{1,2})月(\d{1,2})日", text):
        month_int, day_int = int(month), int(day)
        tokens.extend((
            f"{year}年{month_int}月{day_int}日",
            f"{year}-{month_int:02d}-{day_int:02d}",
            f"{year}{month_int:02d}{day_int:02d}",
        ))
    for year, month, day in re.findall(r"((?:19|20)\d{2})-(\d{1,2})-(\d{1,2})", text):
        month_int, day_int = int(month), int(day)
        tokens.extend((
            f"{year}年{month_int}月{day_int}日",
            f"{year}-{month_int:02d}-{day_int:02d}",
        ))
    return tuple(dict.fromkeys(tokens))


def is_listing_page(url: str) -> bool:
    path = urlsplit(url).path.casefold()
    return path.endswith("/index.html") or bool(re.search(r"-\d+\.html$", path))


def pagination_urls(html: str, page_url: str, *, limit: int, exclude: Iterable[str] = ()) -> tuple[str, ...]:
    """Return sibling numbered pages declared by the listing page itself."""

    if limit <= 0:
        return ()
    directory = page_url.rsplit("/", 1)[0] + "/"
    skipped = {url.rstrip("/") for url in exclude}
    found: list[str] = []
    for name, number in _PAGE_HREF.findall(html):
        candidate = urljoin(directory, f"{name}-{number}.html")
        if candidate.rstrip("/") == page_url.rstrip("/") or candidate.rstrip("/") in skipped:
            continue
        if candidate not in found:
            found.append(candidate)
        if len(found) >= limit:
            break
    return tuple(found)


def target_entries(
    html: str,
    page_url: str,
    tokens: Sequence[str],
    *,
    limit: int = 2,
    window: int = 240,
) -> tuple[ListingEntry, ...]:
    """Return entries whose title or trailing list row matches a question token."""

    if limit <= 0 or not tokens:
        return ()
    entries: list[ListingEntry] = []
    seen: set[str] = set()
    for match in _ANCHOR.finditer(html):
        url = urljoin(page_url, unescape(match.group(1)))
        if not url.startswith(("http://", "https://")):
            continue
        inner = match.group(2)
        title_attr = _TITLE_ATTR.search(match.group(0))
        if title_attr:
            title = unescape(title_attr.group(1))
        else:
            title = unescape(_TAG.sub("", inner))
        title = re.sub(r"\s+", " ", title).strip()
        context = html[match.end():match.end() + window]
        if not any(token in title or token in context or token in match.group(0) for token in tokens):
            continue
        if url in seen:
            continue
        seen.add(url)
        entries.append(ListingEntry(url=url, title=title or url))
        if len(entries) >= limit:
            break
    return tuple(entries)


async def fetch_raw_html(url: str, *, timeout: float = 20.0) -> str:
    """Fetch a page's raw markup; listing links exist only in the HTML."""

    try:
        import aiohttp
    except Exception:  # pragma: no cover - aiohttp is a hard dependency
        return ""
    try:
        from page_fetcher import _ACCEPT_ENCODING, _BROWSER_UA
    except Exception:  # pragma: no cover - defensive default
        _BROWSER_UA = "Mozilla/5.0 (compatible; FinAgent/1.0)"
        _ACCEPT_ENCODING = "gzip, deflate"
    try:
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        async with aiohttp.ClientSession(timeout=client_timeout) as session:
            async with session.get(
                url,
                headers={"User-Agent": _BROWSER_UA, "Accept-Encoding": _ACCEPT_ENCODING},
                allow_redirects=True,
            ) as response:
                if response.status != 200:
                    return ""
                body = await response.read()
    except Exception:
        return ""
    return body.decode("utf-8", errors="replace")


__all__ = [
    "ListingEntry",
    "date_tokens",
    "fetch_raw_html",
    "is_listing_page",
    "pagination_urls",
    "target_entries",
]
