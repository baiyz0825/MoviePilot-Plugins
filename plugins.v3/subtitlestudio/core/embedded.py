"""提取视频内嵌字幕轨。

用 ffprobe 扫描字幕流，ffmpeg 提取文本字幕（SRT/ASS/SSA）。
图形字幕（PGS/VobSub）不提取。STRM 不提取。
按 source_preference 选最佳非目标语言轨，作为翻译源。
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cuegraph import parse_subtitle
from .logging import studio_log
from .models import CueGraph

# 图形字幕，提取了也不能当文本用
SKIP_CODECS = {"pgssub", "dvd_subtitle", "vobsub", "hdmv_pgs_subtitle", "pgs"}


def probe_subtitle_streams(path: str, logger: Any = None) -> List[Dict[str, Any]]:
    """ffprobe 扫描字幕流，返回 [{index, codec, language, title}]，跳过图形字幕。"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-select_streams", "s", path],
            capture_output=True, text=True, timeout=30,
        )
    except FileNotFoundError:
        studio_log(logger, "warning", "未安装 ffprobe，跳过内嵌字幕提取")
        return []
    except Exception as exc:  # noqa: BLE001
        studio_log(logger, "warning", "ffprobe 扫描失败：%s", exc)
        return []
    if result.returncode != 0 or not result.stdout.strip():
        return []
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return []
    streams: List[Dict[str, Any]] = []
    for item in data.get("streams") or []:
        codec = str(item.get("codec_name") or "")
        if codec in SKIP_CODECS:
            continue
        streams.append({
            "index": int(item.get("index") or 0),
            "codec": codec,
            "language": str(item.get("tags", {}).get("language") or ""),
            "title": str(item.get("tags", {}).get("title") or ""),
        })
    return streams


def extract_stream(path: str, stream_index: int, logger: Any = None) -> Optional[str]:
    """ffmpeg 提取指定字幕流为 SRT 文本。"""
    try:
        result = subprocess.run(
            ["ffmpeg", "-v", "quiet", "-i", path,
             "-map", f"0:{stream_index}", "-f", "srt", "-"],
            capture_output=True, text=True, timeout=120,
        )
    except FileNotFoundError:
        studio_log(logger, "warning", "未安装 ffmpeg，跳过内嵌字幕提取")
        return None
    except Exception as exc:  # noqa: BLE001
        studio_log(logger, "warning", "ffmpeg 提取内嵌字幕失败 stream=%s：%s", stream_index, exc)
        return None
    if result.returncode != 0 or not result.stdout.strip():
        return None
    return result.stdout


def _is_target_lang(language: str, target_prefixes: set) -> bool:
    lang = (language or "").lower()
    if not lang:
        return False
    return any(lang.startswith(prefix) for prefix in target_prefixes)


def extract_best_embedded(job: Any, config: Dict[str, Any], logger: Any = None) -> Optional[CueGraph]:
    """按 source_preference 选最佳内嵌文本字幕轨，提取并解析成 CueGraph。

    优先级：先过滤掉目标语言字幕（已经是中文的不用提取翻译），
    再按 english_first / english_only / original_first 选轨。
    """
    path = str(getattr(job, "path", "") or "")
    if not path or getattr(job, "is_strm", False):
        return None
    if not Path(path).is_file():
        return None
    streams = probe_subtitle_streams(path, logger)
    if not streams:
        return None
    target_langs = [str(item) for item in (config.get("target_languages") or ["zh-Hans"])]
    target_prefixes = {lang[:2].lower() for lang in target_langs} | {lang.lower() for lang in target_langs}
    candidates = [s for s in streams if not _is_target_lang(s["language"], target_prefixes)]
    if not candidates:
        studio_log(logger, "info", "内嵌字幕都是目标语言，无需提取翻译 path=%s", path)
        return None
    pref = str(config.get("source_preference") or "english_first")
    if pref == "english_only":
        picked = next((s for s in candidates if "en" in s["language"].lower()), None)
    elif pref == "english_first":
        picked = next((s for s in candidates if "en" in s["language"].lower()), None) or candidates[0]
    else:  # original_first
        picked = candidates[0]
    if not picked:
        studio_log(logger, "info", "没有符合条件的内嵌字幕轨 pref=%s path=%s", pref, path)
        return None
    studio_log(
        logger, "info", "提取内嵌字幕 stream=%s codec=%s lang=%s title=%s",
        picked["index"], picked["codec"], picked["language"], picked["title"],
    )
    content = extract_stream(path, picked["index"], logger)
    if not content:
        return None
    graph = parse_subtitle(
        content,
        f"embedded_{picked['index']}.srt",
        lang=picked["language"] or "source",
        job_id=str(getattr(job, "job_id", "") or ""),
    )
    studio_log(logger, "info", "内嵌字幕提取完成 cues=%s lang=%s", len(graph.cues), picked["language"] or "source")
    return graph
