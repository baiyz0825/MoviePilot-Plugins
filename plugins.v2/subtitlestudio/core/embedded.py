"""提取视频内嵌字幕轨。

用 ffprobe 扫描字幕流，ffmpeg 提取文本字幕（SRT/ASS/SSA）。
图形字幕（PGS/VobSub）不提取。STRM 不提取。
优先复用目标语言轨；没有时按 source_preference 选择翻译源。
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .cuegraph import parse_subtitle
from .language import canonical_language
from .logging import studio_log
from .models import CueGraph

# 图形字幕，提取了也不能当文本用
SKIP_CODECS = {"pgssub", "dvd_subtitle", "vobsub", "hdmv_pgs_subtitle", "pgs"}

# Total-speaker ranks from Ethnologue 2025; unranked tracks follow in original order.
_SPEAKER_RANK = {
    "en": 1,
    "zh-Hans": 2, "zh-Hant": 2,
    "hi": 3,
    "es": 4,
    "ar": 5,
    "fr": 6,
    "bn": 7,
    "pt": 8,
    "ru": 9,
    "ur": 10,
    "id": 11,
    "de": 12,
    "ja": 13,
    "pcm": 14,
    "mr": 16,
    "vi": 17,
    "te": 18,
    "ha": 19,
    "tr": 20,
    "sw": 22,
    "tl": 23,
    "ta": 24,
    "fa": 27,
    "ko": 28,
    "jv": 30,
    "it": 31,
}


def _speaker_rank(language: str) -> Optional[int]:
    return _SPEAKER_RANK.get(canonical_language(language))


class _EmbeddedExtractionTimeout(Exception):
    """Signal that retrying another track would rescan the same slow media file."""


def probe_video_resolution(path: str, logger: Any = None) -> tuple:
    """返回视频显示宽高。失败时 (0, 0)，调用方再退回 1920×1080。"""
    if not path or not Path(path).is_file():
        return 0, 0
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-select_streams", "v:0", path],
            capture_output=True, text=True, timeout=20,
        )
    except FileNotFoundError:
        studio_log(logger, "info", "未安装 ffprobe，ASS 画面按 16:9 写出")
        return 0, 0
    except Exception as exc:  # noqa: BLE001
        studio_log(logger, "warning", "读取视频分辨率失败：%s", exc)
        return 0, 0
    if result.returncode != 0 or not result.stdout.strip():
        return 0, 0
    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError:
        return 0, 0
    streams = data.get("streams") or []
    if not streams:
        return 0, 0
    stream = streams[0]
    try:
        width = int(stream.get("width") or 0)
        height = int(stream.get("height") or 0)
    except (TypeError, ValueError):
        return 0, 0
    sar = str(stream.get("sample_aspect_ratio") or "1:1")
    if ":" in sar:
        num, den = sar.split(":", 1)
        try:
            if int(num) > 0 and int(den) > 0:
                width = int(round(width * int(num) / int(den)))
        except ValueError:
            pass
    tags = stream.get("tags") if isinstance(stream.get("tags"), dict) else {}
    try:
        rotation = abs(int(float(str(tags.get("rotate") or 0)))) % 360
    except (TypeError, ValueError):
        rotation = 0
    if rotation in {90, 270}:
        width, height = height, width
    if width < 16 or height < 16:
        return 0, 0
    return width, height


def probe_subtitle_streams(
    path: str,
    logger: Any = None,
    *,
    include_graphics: bool = False,
) -> List[Dict[str, Any]]:
    """ffprobe 扫描字幕流；默认跳过图形字幕，库存检测可只读其语言元数据。"""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-print_format", "json",
             "-show_streams", "-show_format", "-select_streams", "s", path],
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
    format_data = data.get("format") if isinstance(data.get("format"), dict) else {}
    try:
        media_duration = float(format_data.get("duration") or 0)
    except (TypeError, ValueError):
        media_duration = 0.0
    streams: List[Dict[str, Any]] = []
    for item in data.get("streams") or []:
        codec = str(item.get("codec_name") or "")
        if codec in SKIP_CODECS and not include_graphics:
            continue
        tags = item.get("tags") if isinstance(item.get("tags"), dict) else {}
        disposition = item.get("disposition") if isinstance(item.get("disposition"), dict) else {}
        streams.append({
            "index": int(item.get("index") or 0),
            "codec": codec,
            "language": str(tags.get("language") or ""),
            "title": str(tags.get("title") or ""),
            "duration": str(item.get("duration") or tags.get("DURATION") or ""),
            "media_duration": media_duration,
            "default": bool(disposition.get("default")),
            "forced": bool(disposition.get("forced")),
        })
    return streams


def extract_stream(
    path: str,
    stream_index: int,
    logger: Any = None,
    *,
    codec: str = "",
    context: str = "",
    media_duration: float = 0.0,
) -> Optional[str]:
    """ffmpeg 提取字幕流，定期记录容器扫描进度及最终耗时。"""
    no_progress_timeout_seconds = 300
    progress_interval = 30
    if media_duration > 0:
        timeout_seconds = int(max(300, min(1800, media_duration / 5)))
    else:
        timeout_seconds = 600
    label = context or f"stream={stream_index}"
    read_fd, write_fd = os.pipe()
    command = [
        "ffmpeg", "-hide_banner", "-nostdin", "-v", "error",
        "-progress", f"pipe:{write_fd}", "-nostats", "-i", path,
        "-map", f"0:{stream_index}",
    ]
    if codec.lower() in {"subrip", "srt"}:
        command.extend(["-c:s", "copy"])
    command.extend(["-f", "srt", "-"])
    progress: Dict[str, str] = {}
    progress_lock = threading.Lock()
    progress_activity = {"last_change": time.monotonic()}

    def _read_progress() -> None:
        try:
            with os.fdopen(read_fd, "r", encoding="utf-8", errors="replace") as pipe:
                for line in pipe:
                    key, separator, value = line.strip().partition("=")
                    if separator:
                        with progress_lock:
                            if key in {"out_time", "out_time_ms", "total_size"} and progress.get(key) != value:
                                progress_activity["last_change"] = time.monotonic()
                            progress[key] = value
        except OSError:
            pass

    def _progress_snapshot() -> str:
        with progress_lock:
            values = dict(progress)
        keys = ("out_time", "out_time_ms", "speed", "total_size", "progress")
        fields = [f"{key}={values[key]}" for key in keys if values.get(key)]
        out_time = values.get("out_time", "")
        if media_duration > 0 and out_time and out_time != "N/A":
            try:
                hours, minutes, seconds = out_time.split(":", 2)
                elapsed_media = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
                fields.append(f"media_progress={min(100.0, max(0.0, elapsed_media / media_duration * 100)):.1f}%")
            except (TypeError, ValueError):
                pass
        return " ".join(fields) or "尚无 ffmpeg 进度数据（可能仍在探测或读取输入）"

    def _stderr_detail(stderr: Optional[str]) -> str:
        lines = [line.strip() for line in (stderr or "").splitlines() if line.strip()]
        return " | ".join(lines[-6:])[-1200:]

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            pass_fds=(write_fd,),
        )
    except FileNotFoundError:
        os.close(read_fd)
        os.close(write_fd)
        studio_log(logger, "warning", "未安装 ffmpeg，跳过内嵌字幕提取 %s", label)
        return None
    except Exception as exc:  # noqa: BLE001
        os.close(read_fd)
        os.close(write_fd)
        studio_log(logger, "warning", "启动 ffmpeg 提取内嵌字幕失败 %s：%s", label, exc)
        return None
    os.close(write_fd)
    started = time.monotonic()
    progress_activity["last_change"] = started
    progress_reader = threading.Thread(target=_read_progress, name="subtitle-ffmpeg-progress", daemon=True)
    progress_reader.start()
    last_progress_log = 0.0
    stdout = ""
    stderr = ""
    timed_out = False
    timeout_reason = ""
    studio_log(
        logger, "info",
        "ffmpeg 开始提取内嵌字幕 %s media_duration=%.1fs max_duration=%ss no_progress_timeout=%ss copy_subrip=%s",
        label, media_duration, timeout_seconds, no_progress_timeout_seconds,
        codec.lower() in {"subrip", "srt"},
    )
    try:
        while True:
            elapsed = time.monotonic() - started
            remaining = timeout_seconds - elapsed
            if remaining <= 0:
                timed_out = True
                timeout_reason = "max_duration"
                break
            try:
                stdout, stderr = process.communicate(timeout=min(progress_interval, remaining))
                break
            except subprocess.TimeoutExpired:
                now = time.monotonic()
                elapsed = now - started
                with progress_lock:
                    idle_elapsed = now - progress_activity["last_change"]
                if idle_elapsed >= no_progress_timeout_seconds:
                    timed_out = True
                    timeout_reason = "no_progress"
                    break
                if elapsed >= timeout_seconds:
                    timed_out = True
                    timeout_reason = "max_duration"
                    break
                if elapsed - last_progress_log >= progress_interval - 1:
                    studio_log(
                        logger, "info",
                        "ffmpeg 仍在提取内嵌字幕 %s elapsed=%.1fs idle=%.1fs progress={%s}",
                        label, elapsed, idle_elapsed, _progress_snapshot(),
                    )
                    last_progress_log = elapsed
        if timed_out:
            if process.poll() is not None:
                timed_out = False
                timeout_reason = ""
            else:
                try:
                    process.kill()
                except ProcessLookupError:
                    timed_out = process.poll() is None
                    if not timed_out:
                        timeout_reason = ""
            stdout, stderr = process.communicate()
    except Exception as exc:  # noqa: BLE001
        if process.poll() is None:
            try:
                process.kill()
            except ProcessLookupError:
                pass
        try:
            stdout, stderr = process.communicate()
        except Exception:  # noqa: BLE001
            pass
        elapsed = time.monotonic() - started
        studio_log(
            logger, "warning", "ffmpeg 提取内嵌字幕异常 %s elapsed=%.1fs progress={%s} error=%s stderr=%s",
            label, elapsed, _progress_snapshot(), exc, _stderr_detail(stderr),
        )
        return None
    finally:
        progress_reader.join(timeout=2)

    elapsed = time.monotonic() - started
    detail = _stderr_detail(stderr)
    snapshot = _progress_snapshot()
    if timed_out:
        studio_log(
            logger, "warning",
            "ffmpeg 提取内嵌字幕超时 %s reason=%s elapsed=%.1fs max_duration=%ss no_progress_timeout=%ss progress={%s} stderr=%s",
            label, timeout_reason, elapsed, timeout_seconds, no_progress_timeout_seconds,
            snapshot, detail or "无错误输出",
        )
        raise _EmbeddedExtractionTimeout(timeout_reason)
    if process.returncode != 0:
        studio_log(
            logger, "warning", "ffmpeg 提取内嵌字幕失败 %s returncode=%s elapsed=%.1fs progress={%s} stderr=%s",
            label, process.returncode, elapsed, snapshot, detail or "无错误输出",
        )
        return None
    if not stdout or not stdout.strip():
        studio_log(
            logger, "warning", "ffmpeg 提取内嵌字幕为空 %s elapsed=%.1fs progress={%s} stderr=%s",
            label, elapsed, snapshot, detail or "无错误输出",
        )
        return None
    studio_log(
        logger, "info", "ffmpeg 完成内嵌字幕提取 %s elapsed=%.1fs output_chars=%s progress={%s}",
        label, elapsed, len(stdout), snapshot,
    )
    return stdout


def extract_best_embedded(job: Any, config: Dict[str, Any], logger: Any = None) -> Optional[CueGraph]:
    """优先复用目标语言内嵌轨，否则按 source_preference 选翻译源轨。

    默认回退顺序按总使用人数排列；english_only 和 original_first 可覆盖回退顺序。
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
    target_order = {canonical_language(lang): index for index, lang in enumerate(target_langs)}
    target_candidates = [s for s in streams if canonical_language(s["language"]) in target_order]
    target_candidates.sort(key=lambda stream: (target_order[canonical_language(stream["language"])], stream["index"]))
    target_indexes = {stream["index"] for stream in target_candidates}
    fallback_candidates = [stream for stream in streams if stream["index"] not in target_indexes]
    pref = str(config.get("source_preference") or "english_first")
    if pref == "english_only":
        fallback_candidates = [s for s in fallback_candidates if canonical_language(s["language"]) == "en"]
    elif pref == "english_first":
        fallback_candidates.sort(key=lambda stream: (_speaker_rank(stream["language"]) or 999, stream["index"]))
    # original_first keeps ffprobe order for fallback tracks.
    candidates = target_candidates + fallback_candidates
    if target_candidates:
        studio_log(
            logger, "info", "发现已存在的目标语言内嵌轨，优先复用 langs=%s path=%s",
            [stream["language"] for stream in target_candidates], path,
        )
    if not candidates:
        studio_log(logger, "info", "没有符合条件的内嵌字幕轨 pref=%s path=%s", pref, path)
        return None
    job_id = str(getattr(job, "job_id", "") or "")
    media_duration = float(streams[0].get("media_duration") or 0)
    candidate_rank = {stream["index"]: rank for rank, stream in enumerate(candidates, 1)}
    target_rank = {stream["index"]: rank for rank, stream in enumerate(target_candidates, 1)}
    fallback_rank = {stream["index"]: rank for rank, stream in enumerate(fallback_candidates, 1)}
    track_summary = []
    for stream in streams:
        rank = candidate_rank.get(stream["index"])
        title = str(stream["title"] or "").replace("\r", " ").replace("\n", " ")[:100]
        if stream["index"] in target_rank:
            selection = f"target_language_order={target_rank[stream['index']]} candidate_order={rank}"
        elif pref == "english_only" and canonical_language(stream["language"]) != "en":
            selection = "excluded_by_english_only"
        elif pref == "original_first" and stream["index"] in fallback_rank:
            selection = f"original_order={fallback_rank[stream['index']]} candidate_order={rank}"
        elif stream["index"] in fallback_rank:
            language_rank = _speaker_rank(stream["language"])
            rank_label = f"speaker_rank={language_rank}" if language_rank is not None else "speaker_rank=unranked"
            selection = f"{rank_label} candidate_order={rank}"
        else:
            selection = "not_selected"
        track_summary.append(
            f"#{stream['index']}:{stream['codec']}:{stream['language'] or 'unknown'} "
            f"title={title!r} default={stream['default']} forced={stream['forced']} "
            f"duration={stream['duration'] or 'unknown'} {selection}"
        )
    studio_log(
        logger, "info",
        "内嵌字幕轨清单 job=%s media=%s media_duration=%.1fs target_languages=%s "
        "source_preference=%s tracks=%s",
        job_id, Path(path).name, media_duration, target_langs, pref, "; ".join(track_summary),
    )
    for picked in candidates:
        media_duration = float(picked.get("media_duration") or 0)
        studio_log(
            logger, "info",
            "尝试提取内嵌字幕 job=%s media=%s stream=%s codec=%s lang=%s title=%s "
            "media_duration=%.1fs track_duration=%s default=%s forced=%s",
            job_id, Path(path).name, picked["index"], picked["codec"], picked["language"], picked["title"],
            media_duration, picked.get("duration") or "unknown", picked.get("default", False), picked.get("forced", False),
        )
        context = (
            f"job={job_id} media={Path(path).name} stream={picked['index']} codec={picked['codec']} "
            f"lang={picked['language'] or 'unknown'} title={picked['title'] or 'untitled'}"
        )
        try:
            content = extract_stream(
                path, picked["index"], logger, codec=picked["codec"], context=context,
                media_duration=media_duration,
            )
        except _EmbeddedExtractionTimeout as exc:
            studio_log(
                logger, "warning",
                "内嵌字幕提取因超时停止 job=%s stream=%s reason=%s，不再重扫同一媒体的其他字幕轨",
                job_id, picked["index"], str(exc),
            )
            return None
        if not content:
            continue
        try:
            graph = parse_subtitle(
                content,
                f"embedded_{picked['index']}.srt",
                lang=picked["language"] or "source",
                job_id=str(getattr(job, "job_id", "") or ""),
            )
        except Exception as exc:  # noqa: BLE001
            studio_log(logger, "warning", "解析内嵌字幕失败 stream=%s：%s", picked["index"], exc)
            continue
        if not graph.cues:
            studio_log(logger, "warning", "内嵌字幕没有可用字幕句 stream=%s", picked["index"])
            continue
        studio_log(logger, "info", "内嵌字幕提取完成 cues=%s lang=%s", len(graph.cues), picked["language"] or "source")
        return graph
    studio_log(logger, "warning", "所有候选内嵌字幕轨均提取失败 path=%s", path)
    return None
