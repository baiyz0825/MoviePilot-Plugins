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
    from app.sdk.events import eventmanager, Event as MPEvent
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

        @staticmethod
        def remove_event_listener(*_args, **_kwargs):
            return None

    eventmanager = _Noop()
    MPEvent = Any

try:
    from app.schemas.types import EventType
except Exception:  # noqa: BLE001 — 官方稳定入口；个别宿主再退回 SDK
    try:
        from app.sdk.events import EventType
    except Exception:
        class EventType:
            TransferComplete = "transfer.complete"
            PluginAction = "plugin.action"

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


def schedule_once(func: Callable, *, delay_seconds: int = 3, job_id: str = "ingest_once", plugin_id: str = "SubtitleStudio") -> None:
    try:
        from app.sdk.scheduler import add_plugin_once_job
        add_plugin_once_job(plugin_id, job_id, func, "字幕工坊入库触发", delay_seconds=delay_seconds)
        return
    except Exception:
        pass
    try:
        from app.sdk import scheduler as scheduler_sdk
        scheduler_sdk.add_plugin_once_job(plugin_id, job_id, func, "字幕工坊入库触发", delay_seconds=delay_seconds)
        return
    except Exception:
        func()


def remove_once(job_id: str = "ingest_once", plugin_id: str = "SubtitleStudio") -> None:
    try:
        from app.sdk.scheduler import remove_plugin_once_job
        remove_plugin_once_job(plugin_id, job_id)
        return
    except Exception:
        pass
    try:
        from app.sdk import scheduler as scheduler_sdk
        scheduler_sdk.remove_plugin_once_job(plugin_id, job_id)
    except Exception:
        return


def get_running_plugin(plugin_id: str):
    """只走 SDK PluginManager，按运行实例 ID 取插件。"""
    try:
        from app.sdk.plugins import PluginManager
        manager = PluginManager()
    except Exception:
        return None
    for name in ("get_plugin", "get_running_plugin"):
        getter = getattr(manager, name, None)
        if not getter:
            continue
        try:
            plugin = getter(plugin_id)
        except Exception:
            plugin = None
        if plugin:
            return plugin
    return None


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


def plugin_notification_type():
    """V3 优先 SDK schema，退回宿主 types。"""
    for module_name in ("app.schemas.types", "app.sdk.schema", "app.sdk.schemas"):
        try:
            module = __import__(module_name, fromlist=["NotificationType"])
            enum = getattr(module, "NotificationType", None)
            if enum is not None:
                return getattr(enum, "Plugin", None)
        except Exception:
            continue
    return None


def deliver_notice(plugin, payload: Dict[str, Any]) -> None:
    """走基类 post_message / 宿主通知渠道。"""
    kwargs: Dict[str, Any] = {
        "title": payload.get("title") or "字幕工坊",
        "text": payload.get("text") or "",
    }
    if payload.get("image"):
        kwargs["image"] = payload["image"]
    mtype = plugin_notification_type()
    if mtype is not None:
        kwargs["mtype"] = mtype
    plugin.post_message(**kwargs)


def load_transfer_history(limit: int = 800) -> List[Dict[str, Any]]:
    """V3 只走公开 Oper，不查询宿主内部表结构。"""
    from .core.history import history_object_to_dict
    items = []
    try:
        from app.db.oper.transferhistory import TransferHistoryOper
        listing = getattr(TransferHistoryOper(), "list_by_page", None)
        if listing:
            try:
                items = listing(page=1, count=limit, status=True) or []
            except TypeError:
                items = listing(1, limit) or []
    except Exception:
        items = []
    return [history_object_to_dict(item) for item in items or []]
