"""特效关键词/人物背景的网络检索。

优先使用 Wikimedia Action API 做真正的全文检索，再用 DuckDuckGo
Instant Answer 补充；两者都没有可用摘要时交给 LLM 兜底。
http 回调签名同 host_http：http(method, url, **kwargs) -> dict | str。
"""

from __future__ import annotations

import html
import re
from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import urlencode

from .logging import studio_log

_UA = "MoviePilot-SubtitleStudio/1.0 (https://github.com/baiyz0825/MoviePilot-Plugins)"


def _get_json(http: Callable, url: str, logger: Any = None) -> Optional[Dict[str, Any]]:
    try:
        payload = http("GET", url, timeout=15, headers={"User-Agent": _UA})
    except Exception as exc:  # noqa: BLE001
        # Search misses are normal. Keep diagnostics at debug and omit query text.
        studio_log(logger, "debug", "网络检索 API 请求失败 host=%s：%s", url.split("/", 3)[2], exc)
        return None
    return payload if isinstance(payload, dict) else None


def wikipedia_search(term: str, http: Callable, *, lang: str = "zh", logger: Any = None) -> Tuple[str, bool]:
    """Search Wikipedia through its supported Action API and return a text snippet."""
    params = urlencode({
        "action": "query", "list": "search", "srsearch": term,
        "srlimit": 3, "srprop": "snippet", "format": "json", "utf8": 1,
    })
    payload = _get_json(http, f"https://{lang}.wikipedia.org/w/api.php?{params}", logger)
    results = ((payload or {}).get("query") or {}).get("search") or []
    for result in results[:3]:
        snippet = str(result.get("snippet") or "") if isinstance(result, dict) else ""
        text = html.unescape(re.sub(r"<[^>]+>", " ", snippet)).strip()
        if text:
            return text, True
    return "", False


def duckduckgo_answer(term: str, http: Callable, *, logger: Any = None) -> Tuple[str, bool]:
    """DuckDuckGo Instant Answer API (answer cards, not general web results)."""
    params = urlencode({"q": term, "format": "json", "no_html": 1, "skip_disambig": 1})
    payload = _get_json(http, f"https://api.duckduckgo.com/?{params}", logger)
    if payload:
        text = str(payload.get("AbstractText") or "").strip()
        if text:
            return text, True
        related = payload.get("RelatedTopics") or []
        for item in related[:1]:
            if isinstance(item, dict):
                text = str(item.get("Text") or "").strip()
                if text:
                    return text, True
    return "", False


def web_research(term: str, title: str, http: Callable, *, logger: Any = None) -> Tuple[str, bool]:
    """Search contextual term first, then the bare term; misses fall through quietly."""
    query = f"{term} {title}".strip() if title else term
    queries = list(dict.fromkeys(key for key in (query, term) if key))
    for lang in ("zh", "en"):
        for key in queries:
            text, ok = wikipedia_search(key, http, lang=lang, logger=logger)
            if ok:
                return text, True
    for key in queries:
        text, ok = duckduckgo_answer(key, http, logger=logger)
        if ok:
            return text, True
    return "", False
