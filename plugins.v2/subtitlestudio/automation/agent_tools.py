"""Agent 工具。必须有 name / description / args_schema / async run / get_tool_message。"""

from __future__ import annotations

from typing import Any, List


try:
    from app.agent.tools.base import MoviePilotTool
except Exception:  # noqa: BLE001
    class MoviePilotTool:
        name = ""
        description = ""
        args_schema = {}

        async def run(self, **kwargs) -> str:
            return ""

        def get_tool_message(self, *args, **kwargs):
            return self.description


class _StudioTool(MoviePilotTool):
    def _plugin(self):
        managers = []
        try:
            from app.sdk.plugins import PluginManager as SdkManager
            managers.append(SdkManager)
        except Exception:
            pass
        try:
            from app.core.plugin import PluginManager as CoreManager
            managers.append(CoreManager)
        except Exception:
            pass
        for manager_cls in managers:
            try:
                manager = manager_cls()
                plugin = getattr(manager, "get_plugin", lambda *_: None)("SubtitleStudio")
                if plugin:
                    return plugin
                plugin = getattr(manager, "get_running_plugin", lambda *_: None)("SubtitleStudio")
                if plugin:
                    return plugin
            except Exception:
                continue
        return None

    def get_tool_message(self, *args, **kwargs):
        return self.description


class SubtitleStudioStatusTool(_StudioTool):
    name = "subtitle_studio_status"
    description = "查询字幕工坊队列状态"
    args_schema = {}

    async def run(self, **kwargs) -> str:
        plugin = self._plugin()
        if not plugin:
            return "插件未运行"
        data = plugin.api_status()
        return str(data.get("data") or data)


class SubtitleStudioEnqueueTool(_StudioTool):
    name = "subtitle_studio_enqueue"
    description = "把一部媒体加入字幕工坊队列"
    args_schema = {"path": "媒体文件路径", "title": "标题"}

    async def run(self, path: str = "", title: str = "", **kwargs) -> str:
        plugin = self._plugin()
        if not plugin:
            return "插件未运行"
        data = plugin.api_create_job({"path": path, "title": title, **kwargs})
        return data.get("message") or str(data.get("data") or "")


def get_agent_tools() -> List[type]:
    return [SubtitleStudioStatusTool, SubtitleStudioEnqueueTool]
