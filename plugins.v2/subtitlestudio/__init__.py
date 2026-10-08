"""字幕工坊 V2 宿主入口。

官方钩子都在这个文件，流水线细节在 core / pipeline / packager。
导入期禁止启动任务、访问网络、连数据库。
TransferComplete 必须真正入队，不要学现网 AutoSub V2 的空转。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.plugins import _PluginBase

from .api.routes import StudioApiMixin, build_api_routes
from .automation.actions import WorkflowMixin, build_actions
from .automation.agent_tools import get_agent_tools
from .core.config_schema import (
    CONFIG_PREFIX,
    PLUGIN_NAME,
    build_config_form,
    normalize_plugin_config,
)
from .core.identity import identity_from_v2_ids
from .host import (
    GENERATION,
    EventType,
    event_title,
    eventmanager,
    files_from_event,
    host_logger,
    http_request,
    identity_from_event,
    library_roots,
    load_transfer_history,
    media_extensions,
    plugin_action_type,
    remove_once,
    schedule_once,
)
from .ingest.gates import is_video_path
from .services import StudioServices


class SubtitleStudio(StudioApiMixin, WorkflowMixin, _PluginBase):
    # 插件名称
    plugin_name = PLUGIN_NAME
    # 插件描述
    plugin_desc = "字幕搜索、识别、免费/大模型翻译、特效顶注与工作台预览。独立完成，不委托其它字幕插件。"
    # 插件图标
    plugin_icon = "https://raw.githubusercontent.com/ifsherlock/MoviePilot-Plugins/main/icons/autosubtitles.jpeg"
    # 主题色
    plugin_color = "#0EA5E9"
    # 插件版本：V2 必须是 1.x，禁止和 V3 共用版本号，避免 Release Tag 碰撞
    plugin_version = "1.0.0"
    # 插件作者
    plugin_author = "ifsherlock"
    # 作者主页
    author_url = "https://github.com/ifsherlock"
    # 插件配置项 ID 前缀
    plugin_config_prefix = CONFIG_PREFIX
    # 加载顺序
    plugin_order = 20
    # 可使用的用户级别
    auth_level = 1

    host_generation = GENERATION
    host_logger = host_logger()
    host_http = staticmethod(http_request)
    host_history = staticmethod(load_transfer_history)
    host_library_roots = staticmethod(library_roots)

    def __init__(self):
        super().__init__()
        self._enabled = False
        self._show_sidebar_nav = True
        self._workflow_enabled = True
        self._config: Dict[str, Any] = {}
        self._services: Optional[StudioServices] = None

    def init_plugin(self, config: dict = None):
        # 没有配置也要能打开页面看历史，但不接新活。
        self._config = normalize_plugin_config(config)
        self._enabled = bool(self._config.get("enabled"))
        self._show_sidebar_nav = bool(self._config.get("show_sidebar_nav", True))
        self._workflow_enabled = bool(self._config.get("workflow_enabled", True))
        if self._services:
            self._services.stop()
            self._services = None
        if not self._enabled:
            remove_once()
            return
        self._services = StudioServices(self)
        self._services.start(self._config)
        self.host_logger.info("[SubtitleStudio] V2 服务已启动")

    def current_config(self) -> Dict[str, Any]:
        return dict(self._config or {})

    @property
    def services(self) -> StudioServices:
        if self._services is None:
            self._services = StudioServices(self)
        return self._services

    def get_state(self) -> bool:
        return bool(self._enabled)

    def stop_service(self):
        if self._services:
            self._services.stop()
            self._services = None
        remove_once()
        self.host_logger.info("[SubtitleStudio] V2 服务已停止")

    @staticmethod
    def get_render_mode() -> Tuple[str, str]:
        return "vue", "dist/assets"

    def get_form(self) -> Tuple[List[dict], Dict[str, Any]]:
        return build_config_form()

    def get_page(self) -> List[dict]:
        return []

    def get_api(self) -> List[Dict[str, Any]]:
        return build_api_routes(self)

    def get_sidebar_nav(self) -> List[Dict[str, Any]]:
        if not self.get_state() or not self._show_sidebar_nav:
            return []
        return [{
            "nav_key": "main",
            "title": "字幕工坊",
            "icon": "mdi-subtitles-outline",
            "section": "organize",
            "permission": "manage",
            "order": 20,
        }]

    def get_command(self) -> List[Dict[str, Any]]:
        return [{
            "cmd": "/subtitle_studio_run",
            "event": plugin_action_type(),
            "desc": "执行字幕工坊",
            "category": "插件命令",
            "data": {"action": "subtitle_studio_run"},
        }]

    def get_actions(self) -> List[Dict[str, Any]]:
        return build_actions(self)

    def get_agent_tools(self) -> List[type]:
        return get_agent_tools()

    def get_service(self) -> List[Dict[str, Any]]:
        if not self.get_state() or not self._config.get("queue_offpeak_enabled"):
            return []
        try:
            from apscheduler.triggers.cron import CronTrigger
            trigger = CronTrigger.from_crontab("0 2 * * *")
        except Exception:
            trigger = "cron"
        return [{
            "id": "SubtitleStudio.Offpeak",
            "name": "字幕工坊错峰队列",
            "trigger": trigger,
            "func": self.run_offpeak,
            "kwargs": {},
        }]

    def get_dashboard_meta(self) -> List[Dict[str, str]]:
        return [{"key": "queue", "name": "字幕队列"}]

    def get_dashboard(self, key: str, **kwargs) -> Optional[Tuple]:
        if key != "queue":
            return None
        counts = self.services.store.counts() if self._services else {}
        attrs = {"cols": 12, "md": 6}
        options = {"refresh": 10, "border": True, "title": "字幕工坊队列"}
        elements = [{
            "component": "VCardText",
            "content": [{
                "component": "span",
                "text": f"运行中 {counts.get('running', 0)} · 等待 {counts.get('pending', 0)} · 失败 {counts.get('failed', 0)}",
            }],
        }]
        return attrs, options, elements

    def run_offpeak(self):
        if self._services:
            self._services.scheduler.kick()

    def notify_job(self, job) -> None:
        try:
            self.post_message(
                title="字幕工坊",
                text=f"{job.title} {job.status}",
            )
        except Exception:
            return

    def ingest_watched_file(self, path: str, trigger: str) -> None:
        if not self.get_state():
            return
        if trigger == "watch" and not self._config.get("ingest_on_watch"):
            return
        if trigger == "strm" and not (self._config.get("strm_enabled") and self._config.get("strm_auto_search")):
            return
        if not is_video_path(path):
            return
        self.services.scheduler.enqueue(
            title=Path(path).stem,
            path=path,
            identity={},
            trigger=trigger,
            priority="P1",
            strategy=self._config.get("transfer_strategy") or "search_then_translate",
            config=self._config,
        )

    def kick_queue(self):
        if self._services:
            self._services.scheduler.kick()

    @eventmanager.register(EventType.TransferComplete)
    def listen_transfer_complete(self, event) -> None:
        """整理完成必须真正入队。不要空转再去扫盘。"""
        if not self.get_state() or not self._config.get("ingest_on_event"):
            return
        identity = identity_from_event(event)
        title = event_title(event, "")
        enqueued = 0
        for path in files_from_event(event):
            if Path(path).suffix.lower() not in set(media_extensions()) | {".strm"} and not is_video_path(path):
                continue
            self.services.scheduler.enqueue(
                title=title or Path(path).stem,
                path=path,
                identity=identity or identity_from_v2_ids(),
                trigger="event",
                priority="P1",
                strategy=self._config.get("transfer_strategy") or "search_then_translate",
                config=self._config,
            )
            enqueued += 1
        if enqueued:
            # 事件里只入队，搜站放到 3 秒防抖，避免整理风暴打满字幕站。
            schedule_once(self.kick_queue, delay_seconds=3)

    @eventmanager.register(plugin_action_type())
    def listen_plugin_action(self, event) -> None:
        data = getattr(event, "event_data", None) or {}
        if not isinstance(data, dict):
            data = {}
        if data.get("action") != "subtitle_studio_run":
            return
        self.kick_queue()
