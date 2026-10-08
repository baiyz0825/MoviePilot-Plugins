"""V2 宿主适配。领域代码禁止直接 from app。

V2 合同：
- 基类在 app.plugins
- 设置 app.core.config.settings
- 事件 app.core.event + EventType
- 日志 app.log.logger
- HTTP httpx，不要写 httpx2
- 身份从 tmdbid / doubanid 合成
- 防抖用 DateTrigger + replace_existing，id = SubtitleStudio.ingest_once
- TransferComplete 必须真正入队，不要空转
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

from .core.identity import identity_from_v2_ids
from .core.paths import parse_multiline_paths

try:
    from app.core.config import settings
except Exception:  # noqa: BLE001 — 单测环境没有宿主
    settings = None

try:
    from app.log import logger
except Exception:  # noqa: BLE001
    import logging
    logger = logging.getLogger("subtitlestudio")

try:
    from app.core.event import eventmanager, Event as MPEvent
    from app.schemas.types import EventType
except Exception:  # noqa: BLE001
    class _Noop:
        @staticmethod
        def register(_event_type):
            def decorator(func):
                return func
            return decorator

        @staticmethod
        def add_event_listener(*_args, **_kwargs):
            return None

    eventmanager = _Noop()
    MPEvent = Any

    class EventType:
        TransferComplete = "transfer.complete"
        PluginAction = "plugin.action"

try:
    from app.core.plugin import PluginManager
except Exception:  # noqa: BLE001
    PluginManager = None

try:
    import httpx
except Exception:  # noqa: BLE001
    httpx = None


GENERATION = "v2"


def host_logger():
    return logger


def media_extensions() -> List[str]:
    if settings is not None:
        return list(getattr(settings, "RMT_MEDIAEXT", []) or [])
    return [".mp4", ".mkv", ".ts", ".avi", ".mov", ".m2ts", ".wmv", ".strm"]


def library_roots() -> List[str]:
    if settings is None:
        return []
    roots = getattr(settings, "LIBRARY_PATHS", None) or getattr(settings, "RMT_PATH", None) or []
    if isinstance(roots, str):
        return parse_multiline_paths(roots)
    return [str(item) for item in roots if item]


def proxy_url() -> Optional[str]:
    if settings is None:
        return None
    proxy = getattr(settings, "PROXY", None)
    if isinstance(proxy, dict):
        return proxy.get("https") or proxy.get("http")
    return None


def identity_from_event(event: Any) -> Dict[str, str]:
    data = getattr(event, "event_data", None) or {}
    if not isinstance(data, dict):
        data = {}
    mediainfo = data.get("mediainfo") or data.get("media") or {}
    if not isinstance(mediainfo, dict):
        mediainfo = {
            "tmdbid": getattr(mediainfo, "tmdb_id", None) or getattr(mediainfo, "tmdbid", None),
            "doubanid": getattr(mediainfo, "douban_id", None) or getattr(mediainfo, "doubanid", None),
        }
    return identity_from_v2_ids(mediainfo.get("tmdbid") or mediainfo.get("tmdb_id"), mediainfo.get("doubanid") or mediainfo.get("douban_id"))


def files_from_event(event: Any) -> List[str]:
    data = getattr(event, "event_data", None) or {}
    if not isinstance(data, dict):
        data = {}
    transferinfo = data.get("transferinfo")
    file_list = getattr(transferinfo, "file_list_new", None) if transferinfo is not None else None
    if isinstance(transferinfo, dict):
        file_list = transferinfo.get("file_list_new") or transferinfo.get("file_list")
    return [str(item).strip() for item in (file_list or []) if str(item).strip()]


def event_title(event: Any, fallback: str) -> str:
    data = getattr(event, "event_data", None) or {}
    mediainfo = data.get("mediainfo") if isinstance(data, dict) else None
    if isinstance(mediainfo, dict):
        return str(mediainfo.get("title") or mediainfo.get("name") or fallback)
    return str(getattr(mediainfo, "title", None) or fallback)


def http_request(method: str, url: str, **kwargs: Any) -> Any:
    if httpx is None:
        raise RuntimeError("V2 宿主缺少 httpx")
    timeout = kwargs.pop("timeout", 20)
    headers = kwargs.pop("headers", None)
    json_body = kwargs.pop("json_body", None)
    proxy = proxy_url() if kwargs.pop("use_proxy", False) else None
    with httpx.Client(timeout=timeout, proxy=proxy) as client:
        response = client.request(method, url, headers=headers, json=json_body)
        response.raise_for_status()
        ctype = response.headers.get("content-type") or ""
        if "json" in ctype:
            return response.json()
        return response.text


def schedule_once(func: Callable, *, delay_seconds: int = 3, job_id: str = "SubtitleStudio.ingest_once") -> None:
    """V2 没有 add_plugin_once_job，用 DateTrigger 覆盖同一 id。"""
    try:
        from app.scheduler import Scheduler
        from apscheduler.triggers.date import DateTrigger
    except Exception:  # noqa: BLE001
        func()
        return
    scheduler = getattr(Scheduler(), "scheduler", None) or Scheduler()
    run_date = datetime.now() + timedelta(seconds=delay_seconds)
    add = getattr(scheduler, "add_job", None)
    if not add:
        func()
        return
    add(func, trigger=DateTrigger(run_date=run_date), id=job_id, replace_existing=True, name="字幕工坊入库触发")


def remove_once(job_id: str = "SubtitleStudio.ingest_once") -> None:
    try:
        from app.scheduler import Scheduler
        scheduler = getattr(Scheduler(), "scheduler", None) or Scheduler()
        remove = getattr(scheduler, "remove_job", None)
        if remove:
            remove(job_id)
    except Exception:
        return


def register_transfer_complete(callback: Callable) -> Callable:
    """V2 官方允许方法上的 @eventmanager.register。"""
    return eventmanager.register(EventType.TransferComplete)(callback)


def plugin_action_type():
    return getattr(EventType, "PluginAction", "plugin.action")


def load_transfer_history(limit: int = 400) -> List[Dict[str, Any]]:
    try:
        from app.db.models.transferhistory import TransferHistory
    except Exception:
        return []
    rows = []
    listing = getattr(TransferHistory, "list_by_page", None) or getattr(TransferHistory, "list", None)
    try:
        items = listing(1, limit) if listing else []
    except Exception:
        return []
    for item in items or []:
        rows.append({
            "title": getattr(item, "title", None) or getattr(item, "src", ""),
            "path": getattr(item, "dest", None) or getattr(item, "dest_file", None) or getattr(item, "src", ""),
            "tmdbid": getattr(item, "tmdbid", None),
            "doubanid": getattr(item, "doubanid", None),
            "type": getattr(item, "type", None) or getattr(item, "media_type", None),
            "season": getattr(item, "seasons", None) or getattr(item, "season", None),
            "episode": getattr(item, "episodes", None) or getattr(item, "episode", None),
        })
    return rows
