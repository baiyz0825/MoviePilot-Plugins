"""特效关键词/人物背景的网络检索。

优先级：维基百科 → DuckDuckGo Instant Answer。
拿到摘要返回 (text, True)；拿不到返回 ("", False)，交给 LLM 兜底。
http 回调签名同 host_http：http(method, url, **kwargs) -> dict | str。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple
from urllib.parse import quote

from .logging import studio_log

_UA = "MoviePilot-SubtitleStudio/1.0 (https://github.com/baiyz0825/MoviePilot-Plugins)"


def _get_json(http: Callable, url: str, logger: Any = None) -> Optional[Dict[str, Any]]:
    try:
        payload = http("GET", url, timeout=15, headers={"User-Agent": _UA})
    except Exception as exc:  # noqa: BLE001
        studio_log(logger, "debug", "网络检索请求失败 url=%s：%s", url, exc)
        return None
    return payload if isinstance(payload, dict) else None


def wikipedia_summary(term: str, http: Callable, *, lang: str = "zh", logger: Any = None) -> Tuple[str, bool]:
    """维基百科 REST summary API，返回页面摘要。带片名的词更准。"""
    payload = _get_json(http, f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(term)}", logger)
    if payload:
        text = str(payload.get("extract") or "").strip()
        if text:
            return text, True
    # 中文拿不到试英文
    if lang != "en":
        payload = _get_json(http, f"https://en.wikipedia.org/api/rest_v1/page/summary/{quote(term)}", logger)
        if payload:
            text = str(payload.get("extract") or "").strip()
            if text:
                return text, True
    return "", False


def duckduckgo_answer(term: str, http: Callable, *, logger: Any = None) -> Tuple[str, bool]:
    """DuckDuckGo Instant Answer API，返回 AbstractText。"""
    payload = _get_json(
        http,
        f"https://api.duckduckgo.com/?q={quote(term)}&format=json&no_html=1&skip_disambig=1",
        logger,
    )
    if payload:
        text = str(payload.get("AbstractText") or "").strip()
        if text:
            return text, True
        # 拿不到摘要时看相关主题
        related = payload.get("RelatedTopics") or []
        for item in related[:1]:
            if isinstance(item, dict):
                text = str(item.get("Text") or "").strip()
                if text:
                    return text, True
    return "", False


def web_research(term: str, title: str, http: Callable, *, logger: Any = None) -> Tuple[str, bool]:
    """按优先级检索：维基百科（带片名→裸词）→ DuckDuckGo（带片名→裸词）。"""
    query = f"{term} {title}".strip() if title else term
    for key in (query, term):
        if not key:
            continue
        text, ok = wikipedia_summary(key, http, logger=logger)
        if ok:
            return text, True
    for key in (query, term):
        if not key:
            continue
        text, ok = duckduckgo_answer(key, http, logger=logger)
        if ok:
            return text, True
    return "", False
