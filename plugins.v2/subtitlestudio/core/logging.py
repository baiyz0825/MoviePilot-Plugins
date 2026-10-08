"""统一处理日志。

前缀固定为 [SubtitleStudio]，方便在 MoviePilot 日志里过滤。
步骤号和 AutoSub 一样用 [Step N]，一条任务从入队到落盘能顺着看完。
不要把 API Key、完整 prompt、字幕全文打进日志。
"""

from __future__ import annotations

from typing import Any

PREFIX = "[SubtitleStudio]"


def studio_log(logger: Any, level: str, message: str, *args) -> None:
    """有 logger 才写。level 用 info / warning / error / debug。"""
    if not logger:
        return
    writer = getattr(logger, level, None)
    if not writer:
        return
    writer(f"{PREFIX} {message}", *args)


def studio_step(logger: Any, number: int, message: str, *args) -> None:
    studio_log(logger, "info", f"[Step {number}] {message}", *args)


def studio_done(logger: Any) -> None:
    """任务结束空两行，和现网 AutoSub 一样方便扫日志。"""
    if not logger:
        return
    logger.info("")
    logger.info("")
