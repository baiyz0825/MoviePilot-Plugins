"""入队门禁。覆盖策略是落盘行为，这里只决定建不建任务。"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

CHINESE_LANGS = {"zh", "cn", "zho", "cmn", "yue", "chi", "cht", "zh-hans", "zh-hant"}
CHINESE_COUNTRIES = {"cn", "hk", "tw", "sg", "china", "hong kong", "taiwan", "singapore", "中国", "大陆", "香港", "台湾"}
CHINESE_NAME = re.compile(r"华语|国产|大陆|内地|港剧|台剧|港片|中国")
CHINESE_SUB_NAME = re.compile(
    r"[.\-_](zh|chi|chs|cht|cn|zh-hans|zh-hant|zh-cn|zh-tw|chi&eng|chs&eng)([.\-_]|$)",
    re.I,
)
SUB_EXTS = {".srt", ".ass", ".ssa", ".vtt", ".webvtt"}
VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".ts", ".m2ts", ".mov", ".wmv", ".m4v", ".flv", ".webm"}
STRM_EXT = ".strm"


def is_video_path(path: str) -> bool:
    suffix = Path(path).suffix.lower()
    return suffix in VIDEO_EXTS or suffix == STRM_EXT


def is_strm_path(path: str) -> bool:
    return Path(path).suffix.lower() == STRM_EXT


def looks_chinese_media(meta: Dict[str, Any] | None) -> bool:
    info = meta or {}
    language = str(info.get("language") or info.get("original_language") or "").lower()
    if language in CHINESE_LANGS:
        return True
    country = str(info.get("country") or info.get("origin_country") or "").lower()
    if country in CHINESE_COUNTRIES:
        return True
    title = str(info.get("title") or info.get("name") or "")
    return bool(CHINESE_NAME.search(title))


def existing_chinese_subtitles(media_path: str) -> list:
    directory = Path(media_path).parent
    stem = Path(media_path).stem
    if not directory.is_dir():
        return []
    found = []
    for item in directory.iterdir():
        if item.suffix.lower() not in SUB_EXTS:
            continue
        name = item.name.lower()
        if stem.lower() in name and CHINESE_SUB_NAME.search(name):
            found.append(str(item))
    return found


def file_size_mb(path: str) -> float:
    try:
        return os.path.getsize(path) / (1024 * 1024)
    except OSError:
        return 0.0


def evaluate_gates(config: Dict[str, Any], path: str, meta: Dict[str, Any] | None = None) -> Tuple[bool, str]:
    """返回 (通过, 原因)。STRM 可以搜，但调用方必须禁止 ASR/调轴。"""
    if not path:
        return False, "缺少媒体路径"
    if not is_video_path(path):
        return False, "不是视频或 STRM"
    if config.get("skip_chinese_media") and looks_chinese_media(meta):
        return False, "跳过中文资源"
    if config.get("skip_existing_chinese") and existing_chinese_subtitles(path):
        return False, "已有中字"
    min_mb = float(config.get("min_file_mb") or 0)
    if min_mb and not is_strm_path(path) and 0 < file_size_mb(path) < min_mb:
        return False, "文件过小"
    return True, ""
