"""用户可见错误文案与任务取消信号。

message 禁止堆栈、Cookie、令牌、内部路径；异常详情只进日志。
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlsplit, urlunsplit


class JobCancelled(Exception):
    """用户取消：流水线应立刻停，不写成功。"""


_SECRET_QUERY = re.compile(r"(?i)(token|key|api[_-]?key|authorization|password|secret)=[^&\s]+")
_ABS_PATH = re.compile(r"(/(?:Users|home|var|tmp|opt|media|mnt|Volumes)[^\s\"']+)|([A-Za-z]:\\[^\s\"']+)")
_URL = re.compile(r"https?://[^\s\"']+", re.I)


def sanitize_url(url: str) -> str:
    raw = str(url or "")
    if not raw:
        return ""
    cleaned = _SECRET_QUERY.sub(r"\1=***", raw)
    try:
        parts = urlsplit(cleaned)
        return urlunsplit((parts.scheme, parts.netloc, parts.path, "", ""))
    except Exception:  # noqa: BLE001
        return cleaned[:120]


def user_message(exc: Any = None, *, fallback: str = "操作失败") -> str:
    """把异常压成一句给前端看的话。"""
    if isinstance(exc, JobCancelled):
        return "任务已取消"
    text = str(exc or "").strip()
    if not text:
        return fallback
    text = _SECRET_QUERY.sub(r"\1=***", text)
    text = _URL.sub(lambda m: sanitize_url(m.group(0)) or "[url]", text)
    text = _ABS_PATH.sub("[path]", text)
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 160:
        text = text[:157] + "..."
    lower = text.lower()
    if any(token in lower for token in ("traceback", "file \"", "line ", "authorization")):
        return fallback
    return text or fallback


def looks_like_subtitle(content: str) -> bool:
    body = str(content or "").strip()
    if len(body) < 24:
        return False
    head = body[:4000]
    if "-->" in head:
        return True
    if re.search(r"(?im)^(dialogue|comment):", head):
        return True
    if "WEBVTT" in head.upper()[:64]:
        return True
    if "[Script Info]" in head or "Format:" in head:
        return True
    if re.search(r"(?im)^\d+\s*$", head) and "-->" in body[:8000]:
        return True
    if "<html" in head.lower() or "<!doctype" in head.lower():
        return False
    return False
