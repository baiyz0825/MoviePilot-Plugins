"""Whisper 听写。STRM 调用方不得走进来。导入期不下载模型。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

from ..core.models import Cue, CueGraph


def transcribe(job, config: Dict[str, Any], logger: Any = None) -> Optional[CueGraph]:
    if job.is_strm:
        return None
    path = Path(job.path)
    if not path.is_file():
        return None
    try:
        from faster_whisper import WhisperModel
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger.warning("[SubtitleStudio] 未安装 faster-whisper，跳过 ASR：%s", exc)
        return None
    model_name = str(config.get("whisper_model") or "base")
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
    lang = getattr(info, "language", None) or "en"
    return CueGraph(job_id=job.job_id, source_lang=lang, cues=cues, duration_ms=max((c.end_ms for c in cues), default=0))
