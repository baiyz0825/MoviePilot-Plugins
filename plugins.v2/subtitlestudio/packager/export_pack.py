"""按导出包合同写字幕文件。

语种 × 版式 × 格式 是对白乘法。notes / SDH / ASR 各有开关，不进乘法。
覆盖策略在落盘时执行：跳过 / 备份后替换 / 直接覆盖。
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..core.cuegraph import render_ass, render_srt, render_vtt, stacked_plain
from ..core.models import Cue, CueGraph
from ..core.naming import extra_tracks, export_plan, sanitize_stem
from ..core.style import style_from_config


WriteFn = Callable[[Path, bytes], None]


def _encode(text: str, encoding: str) -> bytes:
    codec = encoding or "utf-8"
    if codec == "utf-8-sig":
        return text.encode("utf-8-sig")
    if codec == "gb18030":
        return text.encode("gb18030", errors="replace")
    return text.encode("utf-8")


def _render_dialogue(graph: CueGraph, plan: Dict[str, Any], style: Dict[str, Any] | None = None) -> str:
    langs = list(plan.get("langs") or [])
    fmt = plan.get("format")
    stack = plan.get("stack") or "main_bottom"
    if fmt == "ass":
        return render_ass(graph, langs, sizes=plan.get("sizes"), stack=stack, include_notes=False, style=style)
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


def _render_notes(graph: CueGraph, plan: Dict[str, Any], style: Dict[str, Any] | None = None) -> str:
    notes_only = CueGraph(job_id=graph.job_id, notes=list(graph.notes), duration_ms=graph.duration_ms)
    return render_ass(notes_only, plan.get("langs") or ["zh-Hans"], include_notes=True, title="notes", style=style)


def _render_sdh(graph: CueGraph, plan: Dict[str, Any]) -> str:
    sdh = CueGraph(
        job_id=graph.job_id,
        cues=[item for item in graph.cues if item.kind == "sdh"] or list(graph.cues),
        duration_ms=graph.duration_ms,
    )
    lang = (plan.get("langs") or ["zh-Hans"])[0]
    return render_srt(sdh, lang, fallback_lang=graph.source_lang)


def render_plan_text(
    graph: CueGraph,
    plan: Dict[str, Any],
    *,
    asr_graph: Optional[CueGraph] = None,
    style: Dict[str, Any] | None = None,
) -> str:
    kind = plan.get("kind") or "dialogue"
    if kind == "notes":
        return _render_notes(graph, plan, style)
    if kind == "sdh":
        return _render_sdh(graph, plan)
    if kind == "asr":
        source = asr_graph or graph
        lang = (plan.get("langs") or [source.source_lang])[0]
        return render_srt(source, lang, fallback_lang=source.source_lang)
    return _render_dialogue(graph, plan, style)


def build_export_manifest(
    config: Dict[str, Any],
    media_path: str,
    *,
    asr_ran: bool = False,
    source_lang: str = "en",
) -> List[Dict[str, Any]]:
    stem = sanitize_stem(media_path)
    return export_plan(config, stem) + extra_tracks(config, stem, asr_ran=asr_ran, source_lang=source_lang)


def write_export_pack(
    config: Dict[str, Any],
    media_path: str,
    graph: CueGraph,
    *,
    asr_graph: Optional[CueGraph] = None,
    asr_ran: bool = False,
    write: Optional[WriteFn] = None,
) -> List[Dict[str, Any]]:
    """把清单落到视频同目录。返回每条文件的写盘结果。"""
    video = Path(media_path)
    directory = video.parent if str(video.parent) not in {".", ""} else Path(".")
    encoding = str(config.get("encoding") or "utf-8")
    policy = str(config.get("overwrite_policy") or "skip")
    results = []
    for plan in build_export_manifest(config, media_path, asr_ran=asr_ran, source_lang=graph.source_lang):
        target = directory / plan["filename"]
        existed = target.exists()
        if existed and policy == "skip":
            results.append({**plan, "path": str(target), "written": False, "reason": "skipped"})
            continue
        if existed and policy == "backup":
            backup = target.with_suffix(target.suffix + ".bak")
            shutil.copy2(target, backup)
        text = render_plan_text(graph, plan, asr_graph=asr_graph, style=style_from_config(config))
        payload = _encode(text, encoding)
        if write:
            write(target, payload)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(payload)
        results.append({**plan, "path": str(target), "written": True, "reason": "ok"})
    return results
