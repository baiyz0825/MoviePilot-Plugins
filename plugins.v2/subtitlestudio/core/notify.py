"""任务结果通知文案。真正推送走宿主 post_message，这里不 import app。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from .models import Job

NOTIFY_EVENTS = ("success", "failed", "skipped", "cancelled")
DEFAULT_NOTIFY_ON = ["success", "failed"]

STATUS_LABELS = {
    "success": "完成",
    "failed": "失败",
    "skipped": "跳过",
    "cancelled": "已取消",
}

TRIGGER_LABELS = {
    "event": "整理入库",
    "watch": "目录监控",
    "strm": "STRM",
    "manual": "手动提交",
    "workflow": "工作流",
    "command": "远程命令",
}


def normalize_notify_on(value: Any) -> List[str]:
    items: Iterable[Any]
    if isinstance(value, str):
        items = [part.strip() for part in value.replace("，", ",").split(",") if part.strip()]
    else:
        items = value or []
    events: List[str] = []
    for item in items:
        event = str(item).strip()
        if event in NOTIFY_EVENTS and event not in events:
            events.append(event)
    return events


def should_notify(config: Dict[str, Any] | None, job: Job) -> bool:
    config = config or {}
    if not config.get("send_notify"):
        return False
    allowed = normalize_notify_on(config.get("notify_on")) or DEFAULT_NOTIFY_ON
    return job.status in allowed


def build_notify_payload(job: Job, config: Dict[str, Any] | None = None) -> Dict[str, str]:
    """给 MoviePilot post_message 用的 title / text / image。"""
    del config
    label = STATUS_LABELS.get(job.status, job.status)
    filename = Path(job.path).name if job.path else ""
    lines = [job.title or filename or job.job_id]
    if filename and filename != lines[0]:
        lines.append(filename)
    if job.status == "success":
        written = [
            str(item.get("filename") or "")
            for item in (job.payload or {}).get("export") or []
            if item.get("written") and item.get("filename")
        ]
        if written:
            shown = written[:4]
            extra = f" 等 {len(written)} 个" if len(written) > 4 else ""
            lines.append("写出：" + "、".join(shown) + extra)
        else:
            lines.append("处理完成，没有新文件写出")
    elif job.error:
        lines.append(job.error)
    trigger = TRIGGER_LABELS.get(job.trigger, job.trigger)
    if trigger:
        lines.append(f"来源：{trigger}")
    image = str((job.payload or {}).get("poster") or (job.payload or {}).get("image") or "")
    return {
        "title": f"字幕工坊 · {label}",
        "text": "\n".join(line for line in lines if line),
        "image": image,
    }
