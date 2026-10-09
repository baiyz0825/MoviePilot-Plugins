"""V3 宿主适配。领域代码禁止直接 from app。

V3 合同：
- 基类 app.sdk.plugin._PluginBase
- 设置 app.sdk.config
- 事件 app.sdk.events
- 日志 app.sdk.logging
- HTTP HTTPX2 / AsyncRequestUtils，禁止 alias_httpx()
- 通用身份必须 media_source + media_id
- 防抖 add_plugin_once_job
- 导入期不要 eventmanager.register
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .core.identity import identity_from_v3_pair
from .core.paths import parse_multiline_paths

try:
    from app.sdk.config import settings
except Exception:  # noqa: BLE001
    settings = None

try:
    from app.sdk.logging import logger
except Exception:  # noqa: BLE001
    import logging
    logger = logging.getLogger("subtitlestudio")

try:
    from app.sdk.events import eventmanager, Event as MPEvent, EventType
except Exception:  # noqa: BLE001
    try:
        from app.sdk.events import eventmanager, Event as MPEvent
        from app.schemas.types import EventType
    except Exception:
        class _Noop:
            @staticmethod
            def register(_event_type):
                def decorator(func):
                    return func
                return decorator

            @staticmethod
            def add_event_listener(*_args, **_kwargs):
                return None

            @staticmethod
            def remove_event_listener(*_args, **_kwargs):
                return None

        eventmanager = _Noop()
        MPEvent = Any

        class EventType:
            TransferComplete = "transfer.complete"
            PluginAction = "plugin.action"

try:
    from app.sdk.plugins import PluginManager
except Exception:  # noqa: BLE001
    PluginManager = None

try:
    import httpx2
except Exception:  # noqa: BLE001
    httpx2 = None


GENERATION = "v3"


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
            "media_source": getattr(mediainfo, "media_source", None),
            "media_id": getattr(mediainfo, "media_id", None),
            "tmdbid": getattr(mediainfo, "tmdb_id", None) or getattr(mediainfo, "tmdbid", None),
            "doubanid": getattr(mediainfo, "douban_id", None) or getattr(mediainfo, "doubanid", None),
        }
    return identity_from_v3_pair(
        mediainfo.get("media_source"),
        mediainfo.get("media_id"),
        mediainfo.get("tmdbid") or mediainfo.get("tmdb_id"),
        mediainfo.get("doubanid") or mediainfo.get("douban_id"),
    )


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
    """同步封装给领域层。禁止 alias_httpx()。"""
    if httpx2 is None:
        raise RuntimeError("V3 宿主缺少 httpx2")
    timeout = kwargs.pop("timeout", 20)
    headers = kwargs.pop("headers", None)
    json_body = kwargs.pop("json_body", None)
    proxy = proxy_url() if kwargs.pop("use_proxy", False) else None
    with httpx2.Client(timeout=timeout, proxy=proxy) as client:
        response = client.request(method, url, headers=headers, json=json_body)
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code}")
        ctype = response.headers.get("content-type") or ""
        if "json" in ctype:
            return response.json()
        return response.text


def schedule_once(func: Callable, *, delay_seconds: int = 3, job_id: str = "ingest_once") -> None:
    try:
        from app.sdk.scheduler import add_plugin_once_job
        add_plugin_once_job("SubtitleStudio", job_id, func, "字幕工坊入库触发", delay_seconds=delay_seconds)
        return
    except Exception:
        pass
    try:
        from app.sdk import scheduler as scheduler_sdk
        scheduler_sdk.add_plugin_once_job("SubtitleStudio", job_id, func, "字幕工坊入库触发", delay_seconds=delay_seconds)
        return
    except Exception:
        func()


def remove_once(job_id: str = "ingest_once") -> None:
    try:
        from app.sdk.scheduler import remove_plugin_once_job
        remove_plugin_once_job("SubtitleStudio", job_id)
        return
    except Exception:
        pass
    try:
        from app.sdk import scheduler as scheduler_sdk
        scheduler_sdk.remove_plugin_once_job("SubtitleStudio", job_id)
    except Exception:
        return


def register_listener(event_type, callback: Callable) -> None:
    add = getattr(eventmanager, "add_event_listener", None)
    if add:
        add(event_type, callback)
        return
    register = getattr(eventmanager, "register", None)
    if register:
        register(event_type)(callback)


def unregister_listener(event_type, callback: Callable) -> None:
    remove = getattr(eventmanager, "remove_event_listener", None)
    if remove:
        remove(event_type, callback)


def plugin_action_type():
    return getattr(EventType, "PluginAction", "plugin.action")


def load_transfer_history(limit: int = 800) -> List[Dict[str, Any]]:
    """V3 整理记录必须自己拿 Session，不能把 None 传给模型。"""
    from .core.history import history_object_to_dict
    items = []
    try:
        from app.db.models.transferhistory import TransferHistory
        from app.db.session import SessionFactory
        session = SessionFactory()
        try:
            items = TransferHistory.list_by_page(session, page=1, count=limit, status=True) or []
        finally:
            session.close()
    except Exception:
        try:
            from app.db.oper.transferhistory import TransferHistoryOper
            oper = TransferHistoryOper()
            listing = getattr(oper, "list_by_page", None)
            items = listing(page=1, count=limit, status=True) if listing else []
        except Exception:
            items = []
    return [history_object_to_dict(item) for item in items or []]
