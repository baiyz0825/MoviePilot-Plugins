"""CueGraph 编解码。

工作台「按时间」和时间轴、挂载预览都读这里的入点/出点。
预览 ASS 必须由当前图现场打包，不要去读已经落盘的旧文件。
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence

from .models import Cue, CueGraph, LANG_FONT_SIZES
from .style import REFERENCE_PLAY_HEIGHT, ass_style_line, normalize_ass_style, style_sizes

SRT_BLOCK = re.compile(
    r"(\d+)\s*\n(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[,.]\d{3})\s*\n([\s\S]*?)(?=\n\n|\Z)",
    re.MULTILINE,
)
CLOCK = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})")
ASS_DIALOGUE = re.compile(r"^(Dialogue|Comment):\s*(.+)$", re.IGNORECASE | re.MULTILINE)
VTT_BLOCK = re.compile(
    r"(?:(\d+)\n)?(\d{2}:\d{2}:\d{2}[.,]\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}[.,]\d{3}).*\n([\s\S]*?)(?=\n\n|\Z)",
    re.MULTILINE,
)
TIMESTAMP_LINE = re.compile(r"^\d+\s*$|^\d{2}:\d{2}:\d{2}[,.]\d{3}")


def clock_to_ms(text: str) -> int:
    match = CLOCK.search(str(text or ""))
    if not match:
        return 0
    hour, minute, second, fraction = match.groups()
    millis = (fraction + "000")[:3]
    return ((int(hour) * 60 + int(minute)) * 60 + int(second)) * 1000 + int(millis)


def ms_to_clock(value: int, comma: bool = True) -> str:
    value = max(0, int(value))
    millis = value % 1000
    total = value // 1000
    second = total % 60
    total //= 60
    minute = total % 60
    hour = total // 60
    sep = "," if comma else "."
    return f"{hour:02d}:{minute:02d}:{second:02d}{sep}{millis:03d}"


def ms_to_ass(value: int) -> str:
    value = max(0, int(value))
    cs = (value % 1000) // 10
    total = value // 1000
    second = total % 60
    total //= 60
    minute = total % 60
    hour = total // 60
    return f"{hour}:{minute:02d}:{second:02d}.{cs:02d}"


def _clean_text(text: str) -> str:
    lines = []
    for line in str(text or "").replace("\r", "").split("\n"):
        stripped = line.strip()
        if not stripped or TIMESTAMP_LINE.match(stripped):
            continue
        lines.append(stripped)
    return "\n".join(lines).strip()


def parse_srt(content: str, *, lang: str = "source", job_id: str = "") -> CueGraph:
    cues: List[Cue] = []
    for index, match in enumerate(SRT_BLOCK.finditer(str(content or "")), start=1):
        text = _clean_text(match.group(4))
        start_ms = clock_to_ms(match.group(2))
        end_ms = clock_to_ms(match.group(3))
        cues.append(Cue(
            cue_id=f"{job_id or 'cue'}-{index}",
            index=index,
            start_ms=start_ms,
            end_ms=end_ms,
            texts={lang: text} if text else {},
        ))
    duration = max((item.end_ms for item in cues), default=0)
    return CueGraph(job_id=job_id, source_lang=lang, cues=cues, duration_ms=duration)


def parse_vtt(content: str, *, lang: str = "source", job_id: str = "") -> CueGraph:
    body = re.sub(r"^WEBVTT.*\n", "", str(content or ""), count=1)
    cues: List[Cue] = []
    for index, match in enumerate(VTT_BLOCK.finditer(body), start=1):
        text = _clean_text(match.group(4))
        cues.append(Cue(
            cue_id=f"{job_id or 'cue'}-{index}",
            index=index,
            start_ms=clock_to_ms(match.group(2)),
            end_ms=clock_to_ms(match.group(3)),
            texts={lang: text} if text else {},
        ))
    return CueGraph(
        job_id=job_id,
        source_lang=lang,
        cues=cues,
        duration_ms=max((item.end_ms for item in cues), default=0),
    )


def _split_ass_fields(payload: str) -> List[str]:
    # ASS Dialogue 前 9 个逗号字段，其余是文本（文本里也可以有逗号）。
    parts = payload.split(",", 9)
    if len(parts) < 10:
        return parts + [""] * (10 - len(parts))
    return parts


def parse_ass(content: str, *, lang: str = "source", job_id: str = "") -> CueGraph:
    cues: List[Cue] = []
    notes: List[Cue] = []
    index = 0
    for match in ASS_DIALOGUE.finditer(str(content or "")):
        fields = _split_ass_fields(match.group(2))
        start_ms = _ass_clock_to_ms(fields[1])
        end_ms = _ass_clock_to_ms(fields[2])
        style = fields[3]
        text = fields[9].replace("\\N", "\n").replace("\\n", "\n")
        text = re.sub(r"\{[^}]*\}", "", text).strip()
        if not text:
            continue
        index += 1
        cue = Cue(
            cue_id=f"{job_id or 'cue'}-{index}",
            index=index,
            start_ms=start_ms,
            end_ms=end_ms,
            texts={lang: text},
            kind="note" if "an8" in fields[9].lower() or "note" in style.lower() else "dialogue",
        )
        if cue.kind == "note":
            notes.append(cue)
        else:
            cues.append(cue)
    return CueGraph(
        job_id=job_id,
        source_lang=lang,
        cues=cues,
        notes=notes,
        duration_ms=max((item.end_ms for item in cues + notes), default=0),
    )


def _ass_clock_to_ms(text: str) -> int:
    match = re.search(r"(\d+):(\d{2}):(\d{2})[.](\d{1,2})", str(text or ""))
    if not match:
        return clock_to_ms(text)
    hour, minute, second, cs = match.groups()
    return ((int(hour) * 60 + int(minute)) * 60 + int(second)) * 1000 + int(cs.ljust(2, "0")) * 10


def parse_subtitle(content: str, filename: str = "", *, lang: str = "source", job_id: str = "") -> CueGraph:
    name = filename.lower()
    body = str(content or "")
    if name.endswith(".ass") or name.endswith(".ssa") or "[script info]" in body.lower():
        return parse_ass(body, lang=lang, job_id=job_id)
    if name.endswith(".vtt") or body.lstrip().startswith("WEBVTT"):
        return parse_vtt(body, lang=lang, job_id=job_id)
    return parse_srt(body, lang=lang, job_id=job_id)


def merge_translation(graph: CueGraph, translations: Dict[str, Dict[str, str]]) -> CueGraph:
    """translations: {cue_id: {lang: text}}。按 id 对齐，不要求数组顺序。"""
    for cue in graph.cues:
        extra = translations.get(cue.cue_id) or {}
        cue.texts.update({str(key): str(value) for key, value in extra.items() if value is not None})
    return graph


def render_srt(graph: CueGraph, lang: str, *, fallback_lang: str = "") -> str:
    blocks = []
    for index, cue in enumerate(graph.sorted_cues(), start=1):
        text = cue.text(lang) or (cue.text(fallback_lang) if fallback_lang else "") or cue.text()
        if not text:
            continue
        blocks.append(
            f"{index}\n{ms_to_clock(cue.start_ms)} --> {ms_to_clock(cue.end_ms)}\n{text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_vtt(graph: CueGraph, lang: str, *, fallback_lang: str = "") -> str:
    lines = ["WEBVTT", ""]
    for index, cue in enumerate(graph.sorted_cues(), start=1):
        text = cue.text(lang) or (cue.text(fallback_lang) if fallback_lang else "") or cue.text()
        if not text:
            continue
        lines.append(str(index))
        lines.append(f"{ms_to_clock(cue.start_ms, comma=False)} --> {ms_to_clock(cue.end_ms, comma=False)}")
        lines.append(text)
        lines.append("")
    return "\n".join(lines)


def script_resolution(display_width: int = 0, display_height: int = 0) -> tuple:
    """虚拟画面，高度固定 1080，宽度跟成片显示比例走。

    Aegisub 的 Script Resolution：字号和边距是这块画面上的坐标，
    不是显示器像素。比例必须和视频一致，16:9 才是 1920×1080。
    """
    height = REFERENCE_PLAY_HEIGHT
    width = int(display_width or 0)
    frame_height = int(display_height or 0)
    if width < 16 or frame_height < 16:
        return 1920, height
    play_x = max(16, int(round(height * width / frame_height)))
    return play_x, height


def render_ass(
    graph: CueGraph,
    langs: Sequence[str],
    *,
    sizes: Sequence[int] | None = None,
    stack: str = "main_bottom",
    include_notes: bool = True,
    title: str = "SubtitleStudio",
    style: Dict[str, Any] | None = None,
    play_width: int = 1920,
    play_height: int = 1080,
) -> str:
    """现场打包预览 / 导出 ASS。

    PlayRes 是整幅虚拟画面，不是把字号锁死在 1080 显示器上。
    字号、边距保持用户填的画面坐标；libass 再把整幅画面缩放到成片。
    LayoutRes 与 PlayRes 相同，这样 720p / 4K 不用各写一套像素。
    """
    spec = normalize_ass_style(style)
    width, height = script_resolution(play_width, play_height)
    used_langs = [item for item in langs if item] or ["source"]
    used_sizes = list(sizes or style_sizes(spec, len(used_langs)) or LANG_FONT_SIZES[: len(used_langs)])
    scaled_sizes = [max(8, int(size)) for size in used_sizes]
    outline_px = max(0, int(spec["outline"]))
    shadow_px = max(0, int(spec["shadow"]))
    margin_v = max(0, int(spec["margin_v"]))
    gap = max(1, int(spec["line_gap"]))
    note_size = max(8, int(spec["note_size"]))
    styles = [
        ass_style_line(
            "Default",
            spec,
            size=scaled_sizes[0],
            color=spec["primary_color"],
            alignment=2,
            margin_v=margin_v,
            outline_px=outline_px,
            shadow_px=shadow_px,
        )
    ]
    for index, size in enumerate(scaled_sizes, start=1):
        # 叠行同侧对齐：main_bottom 全底部（主在下，副递增在上方），main_top 全顶部（主在上，副递增在下方）
        alignment = 2 if stack == "main_bottom" else 8
        line_margin = margin_v + (index - 1) * gap
        styles.append(
            ass_style_line(
                f"Lang{index}",
                spec,
                size=size,
                color=spec["primary_color"],
                alignment=alignment,
                margin_v=line_margin,
                outline_px=outline_px,
                shadow_px=shadow_px,
            )
        )
    styles.append(
        ass_style_line(
            "Note",
            spec,
            size=note_size,
            color=spec["note_color"],
            alignment=8,
            margin_v=max(16, margin_v),
            outline_px=outline_px,
            shadow_px=shadow_px,
        )
    )
    events = ["Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text"]
    for cue in graph.sorted_cues():
        if len(used_langs) == 1:
            text = _ass_escape(cue.text(used_langs[0]) or cue.text())
            if text:
                events.append(_dialogue(cue, "Lang1", text))
        else:
            ordered = list(used_langs)
            if stack == "main_top":
                ordered = list(reversed(ordered))
            for rank, lang in enumerate(ordered, start=1):
                text = _ass_escape(cue.text(lang))
                if not text:
                    continue
                style = f"Lang{used_langs.index(lang) + 1}"
                events.append(_dialogue(cue, style, text, layer=rank))
    if include_notes:
        for cue in graph.notes:
            text = _ass_escape(r"{\an8}" + (cue.text() or ""))
            if text:
                events.append(_dialogue(cue, "Note", text, layer=8))
    header = [
        "[Script Info]",
        f"Title: {title}",
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        f"PlayResX: {width}",
        f"PlayResY: {height}",
        f"LayoutResX: {width}",
        f"LayoutResY: {height}",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, "
        "Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, "
        "Shadow, Alignment, MarginL, MarginR, MarginV, Encoding",
        *styles,
        "",
        "[Events]",
        *events,
        "",
    ]
    return "\n".join(header)


def _ass_escape(text: str) -> str:
    return str(text or "").replace("\n", r"\N")


def _dialogue(cue: Cue, style: str, text: str, layer: int = 0) -> str:
    return (
        f"Dialogue: {layer},{ms_to_ass(cue.start_ms)},{ms_to_ass(cue.end_ms)},{style},,0,0,0,,{text}"
    )


def stacked_plain(cue: Cue, langs: Iterable[str], stack: str = "main_bottom") -> str:
    """SRT/VTT 叠行只能换行，字号交给播放器。"""
    ordered = [lang for lang in langs if cue.text(lang)]
    if stack == "main_bottom":
        ordered = list(reversed(ordered))
    return "\n".join(cue.text(lang) for lang in ordered)
