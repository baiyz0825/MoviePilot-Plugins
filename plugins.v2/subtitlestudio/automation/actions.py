"""工作流动作。func 第一个参数是 ActionContent，返回 (bool, ActionContent)。"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple


def build_actions(plugin) -> List[Dict[str, Any]]:
    if not getattr(plugin, "_workflow_enabled", True):
        return []
    return [
        {"id": "subtitle_query", "name": "查询字幕工坊状态", "func": plugin.wf_query, "kwargs": {}},
        {"id": "subtitle_refresh", "name": "刷新媒体目录", "func": plugin.wf_refresh, "kwargs": {}},
        {"id": "subtitle_match", "name": "在线匹配字幕", "func": plugin.wf_match, "kwargs": {}},
        {"id": "subtitle_generate", "name": "生成字幕", "func": plugin.wf_generate, "kwargs": {}},
        {"id": "subtitle_timeline", "name": "调轴", "func": plugin.wf_timeline, "kwargs": {}},
    ]


def _content_data(content) -> Dict[str, Any]:
    if isinstance(content, dict):
        return content
    data = getattr(content, "data", None) or getattr(content, "action_data", None) or {}
    return data if isinstance(data, dict) else {}


def _ok(content, message: str, data: Dict[str, Any] | None = None) -> Tuple[bool, Any]:
    if hasattr(content, "message"):
        content.message = message
    if data is not None and hasattr(content, "data"):
        content.data = data
    return True, content


class WorkflowMixin:
    def wf_query(self, content) -> Tuple[bool, Any]:
        return _ok(content, "ok", self.api_status().get("data"))

    def wf_refresh(self, content) -> Tuple[bool, Any]:
        return _ok(content, "ok", self.api_refresh_media().get("data"))

    def wf_match(self, content) -> Tuple[bool, Any]:
        data = _content_data(content)
        if data.get("job_id"):
            return _ok(content, "ok", self.api_search_job(data["job_id"]).get("data"))
        job = self.api_create_job({**data, "strategy": "search_only"})
        return _ok(content, job.get("message") or "", job.get("data"))

    def wf_generate(self, content) -> Tuple[bool, Any]:
        data = _content_data(content)
        job = self.api_create_job(data)
        return _ok(content, job.get("message") or "", job.get("data"))

    def wf_timeline(self, content) -> Tuple[bool, Any]:
        return _ok(content, "STRM 和不调轴任务已跳过", {"supported": False})
