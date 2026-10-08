"""Whisper 听写。STRM 调用方不得走进来。导入期不下载模型。"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..core.logging import studio_log
from ..core.models import Cue, CueGraph


def transcribe(job, config: Dict[str, Any], logger: Any = None) -> Optional[CueGraph]:
    if job.is_strm:
        studio_log(logger, "info", "STRM 跳过 ASR path=%s", job.path)
        return None
    path = Path(job.path)
    if not path.is_file():
        studio_log(logger, "warning", "视频文件不存在，跳过 ASR path=%s", job.path)
        return None
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:  # noqa: BLE001
        studio_log(logger, "warning", "未安装 faster-whisper，跳过 ASR：%s", exc)
        return None
    model_name = str(config.get("whisper_model") or "base")
    started = time.monotonic()
    studio_log(logger, "info", "加载 Whisper 模型 %s", model_name)
    model = WhisperModel(model_name, device="auto")
    language = None if config.get("auto_detect_language") else None
    segments, info = model.transcribe(str(path), language=language)
    cues = []
    for index, segment in enumerate(segments, start=1):
        cues.append(Cue(
            cue_id=f"{job.job_id}-asr-{index}",
            index=index,
            start_ms=int((segment.start or 0) * 1000),
            end_ms=int((segment.end or 0) * 1000),
            texts={"source": (segment.text or "").strip()},
            kind="asr",
        ))
        if index % 50 == 0:
            studio_log(logger, "info", "ASR 已识别 %s 段", index)
    lang = getattr(info, "language", None) or "en"
    studio_log(logger, "info", "ASR 识别结束 cues=%s lang=%s 耗时=%.2f秒", len(cues), lang, time.monotonic() - started)
    return CueGraph(job_id=job.job_id, source_lang=lang, cues=cues, duration_ms=max((c.end_ms for c in cues), default=0))
