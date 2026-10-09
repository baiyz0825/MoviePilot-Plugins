"""Agent 工具。必须有 name / description / args_schema / async run / get_tool_message。"""

from __future__ import annotations

from typing import ClassVar, List

from ..host import get_running_plugin

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


try:
    from pydantic import BaseModel, Field

    class EnqueueInput(BaseModel):
        path: str = Field("", description="媒体文件路径")
        title: str = Field("", description="标题")
except Exception:  # noqa: BLE001
    EnqueueInput = {"path": "媒体文件路径", "title": "标题"}


class _StudioTool(MoviePilotTool):
    # MoviePilotTool 是 Pydantic 模型，未标注的类属性会被当成字段，导入即失败。
    plugin_lookup_id: ClassVar[str] = "SubtitleStudio"

    def _plugin(self):
        return get_running_plugin(self.plugin_lookup_id)

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
    args_schema = EnqueueInput

    async def run(self, path: str = "", title: str = "", **kwargs) -> str:
        plugin = self._plugin()
        if not plugin:
            return "插件未运行"
        data = plugin.api_create_job({"path": path, "title": title, **kwargs})
        return data.get("message") or str(data.get("data") or "")


def get_agent_tools(plugin_id: str = "SubtitleStudio") -> List[type]:
    class StatusTool(SubtitleStudioStatusTool):
        plugin_lookup_id: ClassVar[str] = plugin_id

    class EnqueueTool(SubtitleStudioEnqueueTool):
        plugin_lookup_id: ClassVar[str] = plugin_id

    StatusTool.__name__ = f"{plugin_id}StatusTool"
    EnqueueTool.__name__ = f"{plugin_id}EnqueueTool"
    return [StatusTool, EnqueueTool]
