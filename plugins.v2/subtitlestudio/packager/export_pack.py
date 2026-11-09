"""按导出包合同写字幕文件。

语种 × 版式 × 格式 是对白乘法。notes / SDH / ASR 各有开关，不进乘法。
覆盖策略在落盘时执行：跳过 / 备份后替换 / 直接覆盖。
写盘走临时文件 + os.replace，避免中途崩溃截断成品。
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..core.cuegraph import render_ass, render_srt, render_vtt, stacked_plain
from ..core.embedded import probe_video_resolution
from ..core.language import normalize_graph_source_language
from ..core.models import Cue, CueGraph
from ..core.naming import extra_tracks, export_plan, sanitize_stem
from ..core.style import style_from_config


WriteFn = Callable[[Path, bytes], None]
_SUB_SUFFIXES = {".srt", ".ass", ".ssa", ".vtt", ".webvtt"}


def _encode(text: str, encoding: str) -> bytes:
    codec = encoding or "utf-8"
    if codec == "utf-8-sig":
        return text.encode("utf-8-sig")
    if codec == "gb18030":
        return text.encode("gb18030", errors="replace")
    return text.encode("utf-8")


def _atomic_write(target: Path, payload: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        tmp.write_bytes(payload)
        os.replace(tmp, target)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _render_dialogue(
    graph: CueGraph,
    plan: Dict[str, Any],
    style: Dict[str, Any] | None = None,
    *,
    play_width: int = 0,
    play_height: int = 0,
) -> str:
    langs = list(plan.get("langs") or [])
    fmt = plan.get("format")
    stack = plan.get("stack") or "main_bottom"
    if fmt == "ass":
        return render_ass(
            graph,
            langs,
            sizes=plan.get("sizes"),
            stack=stack,
            include_notes=False,
            style=style,
            play_width=play_width,
            play_height=play_height,
        )
    lang = langs[0] if langs else "source"
    if plan.get("layout") == "stacked" and len(langs) > 1:
        clone = CueGraph(
            job_id=graph.job_id,
            source_lang=graph.source_lang,
            cues=[
                Cue(
                    cue_id=item.cue_id,
                    index=item.index,
                    start_ms=item.start_ms,
                    end_ms=item.end_ms,
                    texts={lang: stacked_plain(item, langs, stack)},
                    kind=item.kind,
                )
                for item in graph.sorted_cues()
            ],
            duration_ms=graph.duration_ms,
        )
        return render_srt(clone, lang) if fmt == "srt" else render_vtt(clone, lang)
    return render_srt(graph, lang) if fmt == "srt" else render_vtt(graph, lang)


def _render_notes(
    graph: CueGraph,
    plan: Dict[str, Any],
    style: Dict[str, Any] | None = None,
    *,
    play_width: int = 0,
    play_height: int = 0,
) -> str:
    notes_only = CueGraph(job_id=graph.job_id, notes=list(graph.notes), duration_ms=graph.duration_ms)
    return render_ass(
        notes_only,
        plan.get("langs") or ["zh-Hans"],
        include_notes=True,
        title="notes",
        style=style,
        play_width=play_width,
        play_height=play_height,
    )


def _render_sdh(graph: CueGraph, plan: Dict[str, Any]) -> Optional[str]:
    sdh_cues = [item for item in graph.cues if item.kind == "sdh"]
    if not sdh_cues:
        return None
    sdh = CueGraph(job_id=graph.job_id, cues=sdh_cues, duration_ms=graph.duration_ms)
    lang = (plan.get("langs") or ["zh-Hans"])[0]
    return render_srt(sdh, lang, fallback_lang=graph.source_lang)


def render_plan_text(
    graph: CueGraph,
    plan: Dict[str, Any],
    *,
    asr_graph: Optional[CueGraph] = None,
    style: Dict[str, Any] | None = None,
    play_width: int = 0,
    play_height: int = 0,
) -> Optional[str]:
    graph = normalize_graph_source_language(graph)
    kind = plan.get("kind") or "dialogue"
    if kind == "notes":
        if not graph.notes:
            return None
        return _render_notes(graph, plan, style, play_width=play_width, play_height=play_height)
    if kind == "sdh":
        return _render_sdh(graph, plan)
    if kind == "asr":
        source = asr_graph or graph
        lang = (plan.get("langs") or [source.source_lang])[0]
        return render_srt(source, lang, fallback_lang=source.source_lang)
    return _render_dialogue(graph, plan, style, play_width=play_width, play_height=play_height)


def build_export_manifest(
    config: Dict[str, Any],
    media_path: str,
    *,
    asr_ran: bool = False,
    source_lang: str = "en",
) -> List[Dict[str, Any]]:
    stem = sanitize_stem(media_path)
    return export_plan(config, stem) + extra_tracks(config, stem, asr_ran=asr_ran, source_lang=source_lang)


def _safe_sibling(directory: Path, filename: str) -> Optional[Path]:
    name = str(filename or "")
    if not name or name != Path(name).name or name.startswith("."):
        return None
    root = directory.resolve()
    target = (directory / name).resolve()
    try:
        target.relative_to(root)
    except ValueError:
        return None
    return target


def generated_subtitle_targets(config: Dict[str, Any], media_path: str) -> List[Path]:
    """当前导出设置会写出的字幕，以及它们的 .bak。不含视频，也不含其他外挂。"""
    video = Path(media_path)
    directory = video.parent if str(video.parent) not in {".", ""} else Path(".")
    if not directory.is_dir():
        return []
    video_resolved = video.resolve()
    source_langs = ["en", "zh-Hans", "ja", "ko"]
    forced = str(config.get("force_language") or "").strip()
    if forced:
        source_langs.insert(0, forced)
    names = []
    seen = set()
    for source_lang in source_langs:
        for plan in build_export_manifest(config, media_path, asr_ran=True, source_lang=source_lang):
            filename = str(plan.get("filename") or "")
            if filename in seen:
                continue
            seen.add(filename)
            names.append(filename)
    found: List[Path] = []
    for filename in names:
        target = _safe_sibling(directory, filename)
        if target is None or target == video_resolved or target.suffix.lower() not in _SUB_SUFFIXES:
            continue
        if target.is_file():
            found.append(target)
        backup = _safe_sibling(directory, target.name + ".bak")
        if backup is not None and backup.is_file() and backup.suffix.lower() == ".bak":
            found.append(backup)
    return found


def delete_generated_subtitles(config: Dict[str, Any], media_path: str) -> List[str]:
    """只删除本插件会写出的字幕文件。调用方必须先得到用户确认。"""
    deleted = []
    for path in generated_subtitle_targets(config, media_path):
        try:
            path.unlink()
        except OSError:
            continue
        deleted.append(path.name)
    return deleted


def write_export_pack(
    config: Dict[str, Any],
    media_path: str,
    graph: CueGraph,
    *,
    asr_graph: Optional[CueGraph] = None,
    asr_ran: bool = False,
    write: Optional[WriteFn] = None,
    on_file: Optional[Callable[[int, int, str], None]] = None,
) -> List[Dict[str, Any]]:
    """把清单落到视频同目录。返回每条文件的写盘结果。"""
    video = Path(media_path)
    directory = video.parent if str(video.parent) not in {".", ""} else Path(".")
    encoding = str(config.get("encoding") or "utf-8")
    policy = str(config.get("overwrite_policy") or "skip")
    play_width, play_height = probe_video_resolution(media_path)
    plans = build_export_manifest(config, media_path, asr_ran=asr_ran, source_lang=graph.source_lang)
    results = []
    total = len(plans)
    for index, plan in enumerate(plans, start=1):
        if callable(on_file):
            on_file(index, total, str(plan.get("filename") or ""))
        target = directory / plan["filename"]
        text = render_plan_text(
            graph,
            plan,
            asr_graph=asr_graph,
            style=style_from_config(config),
            play_width=play_width,
            play_height=play_height,
        )
        if text is None:
            results.append({**plan, "path": str(target), "written": False, "reason": "empty"})
            continue
        existed = target.exists()
        if existed and policy == "skip":
            results.append({**plan, "path": str(target), "written": False, "reason": "skipped"})
            continue
        try:
            if existed and policy == "backup":
                backup = target.with_suffix(target.suffix + ".bak")
                shutil.copy2(target, backup)
            payload = _encode(text, encoding)
            if write:
                write(target, payload)
            else:
                _atomic_write(target, payload)
            results.append({**plan, "path": str(target), "written": True, "reason": "ok"})
        except OSError as exc:
            results.append({**plan, "path": str(target), "written": False, "reason": "error", "error": str(exc)})
    return results
