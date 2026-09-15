"""Page fetcher: Jina Reader API (primary) + direct fetch fallback.

Notes:
- Some environments have a brotli/aiohttp incompatibility that breaks decoding
  of `Content-Encoding: br`. To keep fetch robust, we explicitly request only
  `gzip, deflate`.
- PDFs are common in search results (theses, reports). When possible, extract
  text from PDFs for the agent.
"""

import re
import logging
from dataclasses import dataclass
from io import BytesIO

import aiohttp

try:
    from .config import FETCH_TIMEOUT, MAX_PAGE_CHARS
except ImportError:
    from config import FETCH_TIMEOUT, MAX_PAGE_CHARS

logger = logging.getLogger(__name__)

_JINA_PREFIX = "https://r.jina.ai/"

_BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"

# Avoid brotli decoding issues by not requesting br.
_ACCEPT_ENCODING = "gzip, deflate"

try:
    from pypdf import PdfReader  # type: ignore
except Exception:  # pragma: no cover
    PdfReader = None


def _strip_html(html: str) -> str:
    """Remove script/style blocks and HTML tags, return plain text."""
    text = re.sub(r"<script[^>]*>[\s\S]*?</script>", "", html, flags=re.IGNORECASE)
    text = re.sub(r"<style[^>]*>[\s\S]*?</style>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@dataclass(frozen=True)
class RelevantPassage:
    text: str
    start: int
    end: int


def extract_relevant_passages(
    content: str,
    query: str,
    *,
    max_chars: int = MAX_PAGE_CHARS,
    context_chars: int = 700,
) -> list[RelevantPassage]:
    """Select query-relevant windows while retaining offsets and local context."""
    if not isinstance(content, str) or not content.strip() or max_chars <= 0:
        return []
    terms = {
        token.casefold()
        for token in re.findall(r"[A-Za-z0-9][A-Za-z0-9_.%-]{1,}|[\u4e00-\u9fff]{2,}", query)
        if len(token) >= 2
    }
    # Prefer paragraph boundaries. Long unbroken pages are divided into stable
    # windows so a relevant fact near the end is not discarded.
    spans: list[tuple[int, int, str]] = []
    for match in re.finditer(r"\S(?:[\s\S]*?)(?=\n\s*\n|\Z)", content):
        start, end = match.span()
        if end - start <= 2400:
            spans.append((start, end, match.group(0)))
        else:
            for offset in range(start, end, 1800):
                chunk_end = min(end, offset + 2200)
                spans.append((offset, chunk_end, content[offset:chunk_end]))
    if not spans:
        spans = [(0, len(content), content)]

    scored: list[tuple[int, int, int]] = []
    for start, end, text in spans:
        folded = text.casefold()
        hits = sum(1 for term in terms if term in folded)
        phrase_bonus = 4 if query.strip().casefold() in folded else 0
        numeric_bonus = sum(1 for value in re.findall(r"\d+(?:\.\d+)?", query) if value in text)
        scored.append((hits * 3 + phrase_bonus + numeric_bonus, start, end))
    scored.sort(key=lambda row: (-row[0], row[1]))
    chosen: list[tuple[int, int]] = []
    budget = max_chars
    for score, start, end in scored:
        if score <= 0 and chosen:
            break
        window_start = max(0, start - context_chars)
        window_end = min(len(content), end + context_chars)
        if any(not (window_end < old_start or window_start > old_end) for old_start, old_end in chosen):
            continue
        length = window_end - window_start
        if length > budget:
            window_end = window_start + budget
            length = budget
        if length <= 0:
            continue
        chosen.append((window_start, window_end))
        budget -= length
        if budget <= 0:
            break
    if not chosen:
        head = min(len(content), max_chars // 2)
        tail = min(len(content) - head, max_chars - head)
        chosen = [(0, head)]
        if tail > 0:
            chosen.append((len(content) - tail, len(content)))
    chosen.sort()
    return [RelevantPassage(content[start:end].strip(), start, end) for start, end in chosen]


def format_relevant_content(content: str, query: str, *, max_chars: int = MAX_PAGE_CHARS) -> str:
    passages = extract_relevant_passages(content, query, max_chars=max_chars)
    return "\n\n".join(
        f"[relevant passage chars {item.start}-{item.end}]\n{item.text}"
        for item in passages
        if item.text
    )


async def fetch_page_content(url: str) -> str:
    """Fetch a web page and return its text content (max MAX_PAGE_CHARS chars).

    1. Try Jina Reader API (returns clean Markdown).
    2. Fallback: direct aiohttp GET + HTML stripping.
    Returns empty string on total failure.
    """
    is_pdf_url = url.lower().split("?")[0].endswith(".pdf")

    # --- Jina Reader (best effort for HTML) ---
    if not is_pdf_url:
        try:
            jina_url = f"{_JINA_PREFIX}{url}"
            timeout = aiohttp.ClientTimeout(total=FETCH_TIMEOUT)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(
                    jina_url,
                    headers={
                        "User-Agent": _BROWSER_UA,
                        "Accept-Encoding": _ACCEPT_ENCODING,
                    },
                ) as resp:
                    if resp.status == 200:
                        text = await resp.text(errors="replace")
                        text = text.strip()
                        if text and len(text) > 100:
                            logger.debug(f"fetch(jina): got {len(text)} chars from {url[:60]}")
                            return text[:MAX_PAGE_CHARS]
        except Exception as e:
            logger.debug(f"fetch(jina): failed for {url[:60]}: {e}")

    # --- Fallback: direct fetch (HTML/PDF) ---
    try:
        timeout = aiohttp.ClientTimeout(total=FETCH_TIMEOUT)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(
                url,
                headers={
                    "User-Agent": _BROWSER_UA,
                    "Accept-Encoding": _ACCEPT_ENCODING,
                },
                allow_redirects=True,
            ) as resp:
                if resp.status == 200:
                    content_type = (resp.headers.get("Content-Type") or "").lower()
                    is_pdf = is_pdf_url or ("application/pdf" in content_type) or ("pdf" in content_type)

                    if is_pdf:
                        if PdfReader is None:
                            logger.debug("fetch(direct): pypdf not installed; cannot parse PDF")
                            return ""
                        data = await resp.read()
                        reader = PdfReader(BytesIO(data))
                        chunks: list[str] = []
                        total_chars = 0
                        for page in reader.pages:
                            page_text = page.extract_text() or ""
                            if not page_text:
                                continue
                            chunks.append(page_text)
                            total_chars += len(page_text)
                            if total_chars >= MAX_PAGE_CHARS:
                                break
                        text = "\n".join(chunks).strip()
                        if text and len(text) > 100:
                            logger.debug(f"fetch(direct/pdf): got {len(text)} chars from {url[:60]}")
                            return text[:MAX_PAGE_CHARS]
                        return ""

                    html = await resp.text(errors="replace")
                    text = _strip_html(html)
                    if text and len(text) > 100:
                        logger.debug(f"fetch(direct): got {len(text)} chars from {url[:60]}")
                        return text[:MAX_PAGE_CHARS]
    except Exception as e:
        logger.debug(f"fetch(direct): failed for {url[:60]}: {e}")

    return ""
