"""Whisper 听写。STRM 调用方不得走进来。导入期不下载模型。"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

from ..core.logging import studio_log
from ..core.models import Cue, CueGraph

_MODEL_LOCK = threading.Lock()
_MODEL_CACHE: Dict[str, Any] = {}
_MODEL_LAST_USED: Dict[str, float] = {}
_MODEL_TTL_SECONDS = 30 * 60


def _get_whisper_model(model_name: str, logger: Any = None):
    now = time.monotonic()
    with _MODEL_LOCK:
        stale = [key for key, stamp in _MODEL_LAST_USED.items() if now - stamp > _MODEL_TTL_SECONDS and key != model_name]
        for key in stale:
            _MODEL_CACHE.pop(key, None)
            _MODEL_LAST_USED.pop(key, None)
            studio_log(logger, "info", "卸载空闲 Whisper 模型 %s", key)
        cached = _MODEL_CACHE.get(model_name)
        if cached is not None:
            _MODEL_LAST_USED[model_name] = now
            return cached
        from faster_whisper import WhisperModel
        studio_log(logger, "info", "加载 Whisper 模型 %s", model_name)
        model = WhisperModel(model_name, device="auto")
        _MODEL_CACHE[model_name] = model
        _MODEL_LAST_USED[model_name] = now
        return model


def clear_whisper_cache() -> None:
    with _MODEL_LOCK:
        _MODEL_CACHE.clear()
        _MODEL_LAST_USED.clear()


def transcribe(job, config: Dict[str, Any], logger: Any = None, *, cancel_check=None) -> Optional[CueGraph]:
    if job.is_strm:
        studio_log(logger, "info", "STRM 跳过 ASR path=%s", job.path)
        return None
    path = Path(job.path)
    if not path.is_file():
        studio_log(logger, "warning", "视频文件不存在，跳过 ASR path=%s", job.path)
        return None
    try:
        model = _get_whisper_model(str(config.get("whisper_model") or "base"), logger)
    except Exception as exc:  # noqa: BLE001
        studio_log(logger, "warning", "未安装或无法加载 faster-whisper，跳过 ASR：%s", exc)
        return None
    if callable(cancel_check):
        cancel_check()
    started = time.monotonic()
    language = None if config.get("auto_detect_language") else (str(config.get("force_language") or "") or None)
    # auto_detect 开：language=None 让 Whisper 自检；关：若配置了 force_language 则固定，否则仍交给 Whisper
    if not config.get("auto_detect_language") and not language:
        language = None
    segments, info = model.transcribe(str(path), language=language)
    cues = []
    for index, segment in enumerate(segments, start=1):
        if callable(cancel_check) and index % 20 == 0:
            cancel_check()
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
