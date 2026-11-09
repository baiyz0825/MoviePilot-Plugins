"""CueGraph 编解码。

工作台「按时间」和时间轴、挂载预览都读这里的入点/出点。
预览 ASS 必须由当前图现场打包，不要去读已经落盘的旧文件。
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence

from .language import (
    canonical_language,
    detect_text_language,
    is_bilingual_hint,
    normalize_graph_source_language,
    split_bilingual_lines,
)
from .models import Cue, CueGraph, Issue, LANG_FONT_SIZES
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
_ASS_NOTE_STYLE = re.compile(
    r"(?:^|[ _.-])(?:note|sign|effect|title|top)(?:$|[ _.-])|注释|特效|顶部",
    re.IGNORECASE,
)
_ASS_SDH_STYLE = re.compile(r"(?:^|[ _.-])(?:sdh|cc|hearing)(?:$|[ _.-])", re.IGNORECASE)
_ASS_EN_STYLE = re.compile(r"(?:^|[ _.-])(?:en|eng|english|英文)(?:$|[ _.-])", re.IGNORECASE)
_ASS_HANS_STYLE = re.compile(r"(?:^|[ _.-])(?:chs|sc|zh-cn|hans|简中|简体|中文)(?:$|[ _.-])", re.IGNORECASE)
_ASS_HANT_STYLE = re.compile(r"(?:^|[ _.-])(?:cht|tc|zh-tw|hant|繁中|繁体)(?:$|[ _.-])", re.IGNORECASE)
_ASS_JA_STYLE = re.compile(r"(?:^|[ _.-])(?:ja|jpn|japanese|日文|日语)(?:$|[ _.-])", re.IGNORECASE)
_ASS_KO_STYLE = re.compile(r"(?:^|[ _.-])(?:ko|kor|korean|韩文|韩语)(?:$|[ _.-])", re.IGNORECASE)


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


def _texts_and_issues(text: str, lang: str, cue_id: str) -> tuple[Dict[str, str], List[Issue]]:
    split, uncertain = split_bilingual_lines(text, lang)
    if split:
        return split, []
    issues = []
    if uncertain:
        issues.append(Issue(
            code="bilingual_split_uncertain",
            message="检测到混合语言，但无法可靠按行拆轨；已保留原文",
            cue_id=cue_id,
        ))
    key = canonical_language(lang)
    return ({key: text} if text else {}), issues


def parse_srt(content: str, *, lang: str = "source", job_id: str = "") -> CueGraph:
    cues: List[Cue] = []
    for index, match in enumerate(SRT_BLOCK.finditer(str(content or "")), start=1):
        text = _clean_text(match.group(4))
        start_ms = clock_to_ms(match.group(2))
        end_ms = clock_to_ms(match.group(3))
        cue_id = f"{job_id or 'cue'}-{index}"
        texts, issues = _texts_and_issues(text, lang, cue_id)
        cues.append(Cue(
            cue_id=cue_id,
            index=index,
            start_ms=start_ms,
            end_ms=end_ms,
            texts=texts,
            issues=issues,
        ))
    duration = max((item.end_ms for item in cues), default=0)
    graph = CueGraph(job_id=job_id, source_lang=canonical_language(lang), cues=cues, duration_ms=duration)
    return normalize_graph_source_language(graph)


def parse_vtt(content: str, *, lang: str = "source", job_id: str = "") -> CueGraph:
    body = re.sub(r"^WEBVTT.*\n", "", str(content or ""), count=1)
    cues: List[Cue] = []
    for index, match in enumerate(VTT_BLOCK.finditer(body), start=1):
        text = _clean_text(match.group(4))
        cue_id = f"{job_id or 'cue'}-{index}"
        texts, issues = _texts_and_issues(text, lang, cue_id)
        cues.append(Cue(
            cue_id=cue_id,
            index=index,
            start_ms=clock_to_ms(match.group(2)),
            end_ms=clock_to_ms(match.group(3)),
            texts=texts,
            issues=issues,
        ))
    graph = CueGraph(
        job_id=job_id,
        source_lang=canonical_language(lang),
        cues=cues,
        duration_ms=max((item.end_ms for item in cues), default=0),
    )
    return normalize_graph_source_language(graph)


def _split_ass_fields(payload: str) -> List[str]:
    # ASS Dialogue 前 9 个逗号字段，其余是文本（文本里也可以有逗号）。
    parts = payload.split(",", 9)
    if len(parts) < 10:
        return parts + [""] * (10 - len(parts))
    return parts


def _ass_style_language(style: str) -> str:
    value = str(style or "")
    if _ASS_EN_STYLE.search(value):
        return "en"
    if _ASS_HANT_STYLE.search(value):
        return "zh-Hant"
    if _ASS_HANS_STYLE.search(value):
        return "zh-Hans"
    if _ASS_JA_STYLE.search(value):
        return "ja"
    if _ASS_KO_STYLE.search(value):
        return "ko"
    return "source"


def _ass_kind(event_type: str, style: str, raw_text: str) -> str:
    lowered = str(raw_text or "").lower()
    if event_type.lower() == "comment" or "an8" in lowered or _ASS_NOTE_STYLE.search(style or ""):
        return "note"
    if _ASS_SDH_STYLE.search(style or ""):
        return "sdh"
    return "dialogue"


def _ass_texts(text: str, lang: str, style: str, cue_id: str) -> tuple[Dict[str, str], List[Issue]]:
    style_lang = _ass_style_language(style)
    hint = style_lang if style_lang != "source" else lang
    split, uncertain = split_bilingual_lines(text, hint)
    if split:
        return split, []
    detected = detect_text_language(text, hint)
    key = detected if detected != "source" else canonical_language(hint)
    issues = []
    if uncertain or (is_bilingual_hint(lang) and key == "source"):
        issues.append(Issue(
            code="bilingual_split_uncertain",
            message="检测到混合语言 ASS 对白，但无法可靠拆轨；已保留原文",
            cue_id=cue_id,
        ))
    return {key: text}, issues


def _ass_overlap_ratio(left: Cue, right: Cue) -> float:
    overlap = max(0, min(left.end_ms, right.end_ms) - max(left.start_ms, right.start_ms))
    longest = max(1, left.end_ms - left.start_ms, right.end_ms - right.start_ms)
    return overlap / longest


def _can_merge_ass(record: Dict[str, Any], cue: Cue, speaker: str, style: str) -> bool:
    existing = record["cue"]
    if existing.kind != cue.kind or existing.kind not in {"dialogue", "sdh"}:
        return False
    left_speaker = str(record.get("speaker") or "").strip().casefold()
    right_speaker = str(speaker or "").strip().casefold()
    if bool(left_speaker) != bool(right_speaker):
        return False
    if left_speaker and left_speaker != right_speaker:
        return False
    left_langs = {canonical_language(key) for key, value in existing.texts.items() if value}
    right_langs = {canonical_language(key) for key, value in cue.texts.items() if value}
    if not left_langs or not right_langs or "source" in left_langs | right_langs:
        return False
    if left_langs & right_langs or len(left_langs | right_langs) > 2:
        return False
    exact = existing.start_ms == cue.start_ms and existing.end_ms == cue.end_ms
    if not exact:
        if abs(existing.start_ms - cue.start_ms) > 350 or abs(existing.end_ms - cue.end_ms) > 350:
            return False
        if _ass_overlap_ratio(existing, cue) < 0.88:
            return False
        # Near-overlap without actor names needs language-specific styles; this
        # avoids merging unrelated simultaneous dialogue from two characters.
        if not (left_speaker and left_speaker == right_speaker):
            left_style_lang = _ass_style_language(str(record.get("style") or ""))
            right_style_lang = _ass_style_language(style)
            if "source" in {left_style_lang, right_style_lang} or left_style_lang == right_style_lang:
                return False
    return True


def parse_ass(content: str, *, lang: str = "source", job_id: str = "") -> CueGraph:
    records: List[Dict[str, Any]] = []
    notes: List[Cue] = []
    event_index = 0
    for match in ASS_DIALOGUE.finditer(str(content or "")):
        fields = _split_ass_fields(match.group(2))
        start_ms = _ass_clock_to_ms(fields[1])
        end_ms = _ass_clock_to_ms(fields[2])
        style = fields[3]
        speaker = fields[4]
        raw_text = fields[9]
        text = raw_text.replace("\\N", "\n").replace("\\n", "\n")
        text = re.sub(r"\{[^}]*\}", "", text).strip()
        if not text:
            continue
        event_index += 1
        cue_id = f"{job_id or 'cue'}-{event_index}"
        texts, issues = _ass_texts(text, lang, style, cue_id)
        cue = Cue(
            cue_id=cue_id,
            index=event_index,
            start_ms=start_ms,
            end_ms=end_ms,
            texts=texts,
            speaker=speaker,
            kind=_ass_kind(match.group(1), style, raw_text),
            issues=issues,
        )
        if cue.kind == "note":
            notes.append(cue)
            continue
        merged = next((record for record in reversed(records) if _can_merge_ass(record, cue, speaker, style)), None)
        if merged:
            existing = merged["cue"]
            existing.texts.update(cue.texts)
            existing.start_ms = min(existing.start_ms, cue.start_ms)
            existing.end_ms = max(existing.end_ms, cue.end_ms)
            for issue in cue.issues:
                issue.cue_id = existing.cue_id
            existing.issues.extend(cue.issues)
        else:
            records.append({"cue": cue, "speaker": speaker, "style": style})

    cues = [record["cue"] for record in records]
    for index, cue in enumerate(sorted(cues, key=lambda item: (item.start_ms, item.index)), start=1):
        cue.index = index
    graph = CueGraph(
        job_id=job_id,
        source_lang=canonical_language(lang),
        cues=cues,
        notes=notes,
        duration_ms=max((item.end_ms for item in cues + notes), default=0),
    )
    return normalize_graph_source_language(graph)


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
    requested = canonical_language(lang)
    for index, cue in enumerate(graph.sorted_cues(), start=1):
        text = cue.text(requested)
        if not text and fallback_lang:
            text = cue.text(canonical_language(fallback_lang))
        if not text and requested == "source":
            text = cue.text(graph.source_lang) or cue.text("source") or cue.text()
        if not text:
            continue
        blocks.append(
            f"{index}\n{ms_to_clock(cue.start_ms)} --> {ms_to_clock(cue.end_ms)}\n{text}"
        )
    return "\n\n".join(blocks) + ("\n" if blocks else "")


def render_vtt(graph: CueGraph, lang: str, *, fallback_lang: str = "") -> str:
    lines = ["WEBVTT", ""]
    requested = canonical_language(lang)
    for index, cue in enumerate(graph.sorted_cues(), start=1):
        text = cue.text(requested)
        if not text and fallback_lang:
            text = cue.text(canonical_language(fallback_lang))
        if not text and requested == "source":
            text = cue.text(graph.source_lang) or cue.text("source") or cue.text()
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
    used_langs = list(dict.fromkeys(canonical_language(item) for item in langs if item)) or ["source"]
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
        # 两种顺序都固定在画面底部，只交换主副语言的上下关系。
        alignment = 2
        line_margin = margin_v + ((len(scaled_sizes) - index) if stack == "main_top" else (index - 1)) * gap
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
            requested = used_langs[0]
            raw_text = cue.text(requested)
            if not raw_text and requested == "source":
                raw_text = cue.text(graph.source_lang) or cue.text("source") or cue.text()
            text = _ass_escape(raw_text)
            if text:
                events.append(_dialogue(cue, "Lang1", text))
        else:
            ordered = list(used_langs)
            if stack == "main_top":
                ordered = list(reversed(ordered))
            seen_texts = set()
            for rank, lang in enumerate(ordered, start=1):
                raw_text = cue.text(lang)
                identity = re.sub(r"\s+", "", raw_text).casefold()
                if not raw_text or identity in seen_texts:
                    continue
                seen_texts.add(identity)
                text = _ass_escape(raw_text)
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
    """SRT/VTT 叠行只能换行，字号交给播放器；重复文本仅作末端防御。"""
    ordered = [canonical_language(lang) for lang in langs if cue.text(canonical_language(lang))]
    if stack == "main_bottom":
        ordered = list(reversed(ordered))
    lines = []
    seen = set()
    for lang in ordered:
        text = cue.text(lang)
        identity = re.sub(r"\s+", "", text).casefold()
        if not text or identity in seen:
            continue
        seen.add(identity)
        lines.append(text)
    return "\n".join(lines)
