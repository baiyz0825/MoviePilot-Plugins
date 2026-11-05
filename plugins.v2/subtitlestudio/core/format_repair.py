"""模型 / ASR 输出格式修复。

修的是格式，不是再创作：剥壳、抽 JSON、按 id 重排、空句和倒序轴。
现网 AutoSub 行数不对或 id 乱序就整批扔；这里先修再降级，好句先留。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .models import Cue, CueGraph, Issue

THINK_BLOCK = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)
FENCE = re.compile(r"```(?:json|javascript|txt)?\s*([\s\S]*?)```", re.IGNORECASE)
LEADING_CHAT = re.compile(
    r"^(好的|当然|如下所示|下面是|here(?:'|’)s|sure[,:]?|the json(?: is)?[:：]?)[\s\S]{0,80}?\n",
    re.IGNORECASE,
)
TRAILING_COMMA = re.compile(r",(\s*[}\]])")
NUMBERED = re.compile(r"^\s*(?:[-*]|\d+[.)、])\s*(.+)$")
HALLUCINATION = re.compile(r"^(嗯+|啊+|哦+|呃+|um+|uh+|you know)+$", re.IGNORECASE)


def strip_shell(text: str) -> str:
    body = THINK_BLOCK.sub("", str(text or ""))
    fenced = FENCE.findall(body)
    if fenced:
        body = fenced[-1]
    body = LEADING_CHAT.sub("", body.strip())
    return body.strip()


def extract_json(text: str) -> Any:
    """从杂文里取数组或对象。修尾逗号；对象包一层也收下。"""
    body = strip_shell(text)
    candidates = [body]
    start = body.find("[")
    end = body.rfind("]")
    if start != -1 and end > start:
        candidates.append(body[start:end + 1])
    obj_start = body.find("{")
    obj_end = body.rfind("}")
    if obj_start != -1 and obj_end > obj_start:
        candidates.append(body[obj_start:obj_end + 1])
    last_error = None
    for item in candidates:
        try:
            return json.loads(TRAILING_COMMA.sub(r"\1", item))
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
    if last_error:
        raise last_error
    raise json.JSONDecodeError("empty", body, 0)


def _row_id(row: Any, fallback: str) -> str:
    if isinstance(row, dict):
        for key in ("id", "cue_id", "index", "idx"):
            if row.get(key) not in (None, ""):
                return str(row[key])
    return fallback


def _row_text(row: Any) -> str:
    if isinstance(row, str):
        return row.strip()
    if not isinstance(row, dict):
        return str(row or "").strip()
    for key in ("zh", "text", "translation", "target", "src", "en"):
        if row.get(key):
            return str(row[key]).strip()
    return ""


def align_by_id(
    payload: Any,
    expected_ids: Sequence[str],
    *,
    partial_accept: bool = True,
) -> Tuple[Dict[str, str], List[str]]:
    """不要求数组顺序；多的丢掉，缺的记下。"""
    rows: List[Any]
    if isinstance(payload, dict):
        if isinstance(payload.get("items"), list):
            rows = payload["items"]
        elif isinstance(payload.get("translations"), list):
            rows = payload["translations"]
        else:
            rows = [{"id": key, "text": value} for key, value in payload.items()]
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []

    mapped: Dict[str, str] = {}
    expected = [str(item) for item in expected_ids]
    expected_set = set(expected)
    for index, row in enumerate(rows):
        cue_id = _row_id(row, expected[index] if index < len(expected) else "")
        text = _row_text(row)
        if cue_id in expected_set and text:
            mapped[cue_id] = text
    missing = [item for item in expected if item not in mapped]
    if not partial_accept and missing:
        return {}, expected
    return mapped, missing


def fallback_numbered(text: str, expected_ids: Sequence[str]) -> Tuple[Dict[str, str], List[str]]:
    """不是 JSON 时按编号 / 换行试拆，能对齐就用。"""
    lines = []
    for raw in strip_shell(text).splitlines():
        match = NUMBERED.match(raw)
        line = (match.group(1) if match else raw).strip()
        if line:
            lines.append(line)
    mapped = {}
    for cue_id, line in zip(expected_ids, lines):
        mapped[str(cue_id)] = line
    missing = [item for item in expected_ids if item not in mapped]
    if len(lines) != len(expected_ids):
        return mapped, missing or list(expected_ids[len(lines):])
    return mapped, missing


def repair_model_batch(
    raw: str,
    expected_ids: Sequence[str],
    *,
    partial_accept: bool = True,
) -> Tuple[Dict[str, str], List[str], List[str]]:
    """返回 (对齐结果, 缺失 id, 修复步骤)。"""
    steps: List[str] = []
    body = strip_shell(raw)
    steps.append("strip_shell")
    try:
        payload = extract_json(body)
        steps.append("extract_json")
        mapped, missing = align_by_id(payload, expected_ids, partial_accept=partial_accept)
        steps.append("align_by_id")
        # JSON 已经抽出就不要再按行号硬拆，否则缺 id 会被上一句译文填上。
        return mapped, missing, steps
    except (json.JSONDecodeError, ValueError, TypeError):
        steps.append("json_failed")
    mapped, missing = fallback_numbered(body, expected_ids)
    steps.append("numbered_fallback")
    return mapped, missing, steps


def looks_empty_or_hallucination(text: str) -> bool:
    body = re.sub(r"\s+", "", str(text or ""))
    if not body:
        return True
    return bool(HALLUCINATION.match(str(text or "").strip()))


_ABBREV = re.compile(r"(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|vs|etc)\.$", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.!?。！？…]$")
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？!?])\s*|(?<=\.)\s+(?=[A-Z])")
_CLAUSE_SPLIT = re.compile(r"(?<=[，,、；;])\s*")
_PAUSE_MS = 400


def ends_sentence(text: str) -> bool:
    body = str(text or "").strip().strip("\"'”’」』")
    if not body or _ABBREV.search(body):
        return False
    return bool(_SENTENCE_END.search(body))


def split_sentences(text: str) -> List[str]:
    """按句号切开。英文缩写和小数不在这里处理，中文句号后面可以没有空格。"""
    body = str(text or "").strip()
    if not body:
        return []
    parts = [item.strip() for item in _SENTENCE_SPLIT.split(body) if item and item.strip()]
    return parts or [body]


def _split_clauses(text: str) -> List[str]:
    parts = [item.strip() for item in _CLAUSE_SPLIT.split(str(text or "")) if item and item.strip()]
    return parts if len(parts) > 1 else [str(text or "").strip()]


def translation_keeps_cue(source: str, translated: str) -> bool:
    """一句原文只能落成一句译文。把上下文并进来的长段丢掉，留给更小的批次重试。"""
    src = str(source or "").strip()
    dst = str(translated or "").strip()
    if not dst:
        return False
    src_breaks = len(re.findall(r"[.!?。！？]", src))
    dst_breaks = len(re.findall(r"[.!?。！？]", dst))
    if dst_breaks >= src_breaks + 2 and len(dst) > max(48, len(src) * 2):
        return False
    return True


def cues_from_words(words: Sequence[Dict[str, Any]], *, max_segment_ms: int) -> List[Dict[str, Any]]:
    """按停顿和句号把词收成字幕。一段不超过 max_segment_ms。"""
    rows = [item for item in words if str(item.get("text") or "").strip()]
    if not rows:
        return []
    limit = max(1000, int(max_segment_ms or 8000))
    groups: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []

    def flush() -> None:
        nonlocal current
        if current:
            groups.append(current)
            current = []

    for word in rows:
        if current:
            gap = int(word.get("start_ms") or 0) - int(current[-1].get("end_ms") or 0)
            duration = int(word.get("end_ms") or 0) - int(current[0].get("start_ms") or 0)
            previous = "".join(str(item.get("text") or "") for item in current).strip()
            # Whisper can omit punctuation across several spoken sentences.
            # Keep each cue short while retaining actual word boundaries.
            if gap >= _PAUSE_MS or ends_sentence(previous) or duration > min(limit, 4500) or len(previous) + len(str(word.get("text") or "")) > 56:
                flush()
        current.append(word)
    flush()
    cues = []
    for group in groups:
        text = "".join(str(item.get("text") or "") for item in group).strip()
        if not text:
            continue
        cues.append({
            "start_ms": int(group[0].get("start_ms") or 0),
            "end_ms": max(int(group[-1].get("end_ms") or 0), int(group[0].get("start_ms") or 0)),
            "text": text,
        })
    return cues


def _allocate_spans(start: int, end: int, weights: Sequence[int]) -> List[Tuple[int, int]]:
    start = int(start)
    end = max(start, int(end))
    usable = [max(1, int(weight)) for weight in weights]
    total = sum(usable) or 1
    cursor = start
    spans: List[Tuple[int, int]] = []
    for index, weight in enumerate(usable):
        if index == len(usable) - 1:
            spans.append((cursor, end))
            break
        piece = max(1, int(round((end - start) * weight / total)))
        next_cursor = min(end, cursor + piece)
        if next_cursor <= cursor:
            next_cursor = min(end, cursor + 1)
        spans.append((cursor, next_cursor))
        cursor = next_cursor
    return spans


def _split_cue_pieces(cue: Cue, max_segment_ms: int) -> List[Tuple[int, int, str]]:
    """没有词时间时，按句号（过长再按逗号）把一条字幕拆回多句，时间按字数分。"""
    text = cue.text().strip()
    if not text or len(cue.texts) > 1:
        return [(cue.start_ms, cue.end_ms, text)] if text else []
    parts = split_sentences(text)
    limit = max(1000, int(max_segment_ms or 8000))
    if len(parts) == 1 and cue.duration_ms() > limit:
        parts = _split_clauses(text)
    if len(parts) <= 1:
        return [(cue.start_ms, cue.end_ms, text)]
    spans = _allocate_spans(cue.start_ms, cue.end_ms, [len(part) for part in parts])
    pieces: List[Tuple[int, int, str]] = []
    for (start, end), part in zip(spans, parts):
        clauses = _split_clauses(part) if end - start > limit else [part]
        if len(clauses) <= 1:
            pieces.append((start, end, part))
            continue
        for (sub_start, sub_end), clause in zip(_allocate_spans(start, end, [len(item) for item in clauses]), clauses):
            pieces.append((sub_start, sub_end, clause))
    return [(start, end, part) for start, end, part in pieces if part]


def repair_asr_graph(graph: CueGraph, *, enabled: bool = True, max_segment_seconds: float = 8) -> CueGraph:
    """丢掉空句/幻觉，重叠裁开，倒序对调，过长按 CPS 切开。"""
    if not enabled:
        return graph
    repaired: List[Cue] = []
    last_end = 0
    next_index = 1
    max_segment_ms = int(max(1, float(max_segment_seconds or 8)) * 1000)
    for cue in graph.sorted_cues():
        text = cue.text()
        if looks_empty_or_hallucination(text):
            cue.issues.append(Issue(code="asr_empty", message="空句或幻觉已丢弃", cue_id=cue.cue_id, severity="info"))
            continue
        start_ms, end_ms = cue.start_ms, cue.end_ms
        if end_ms < start_ms:
            start_ms, end_ms = end_ms, start_ms
            cue.issues.append(Issue(code="asr_reversed", message="起止时间已对调", cue_id=cue.cue_id))
        if start_ms < last_end:
            start_ms = last_end + 40
            cue.issues.append(Issue(code="asr_overlap", message="重叠轴已裁开", cue_id=cue.cue_id))
        if end_ms <= start_ms:
            end_ms = start_ms + 400
        cue.start_ms = start_ms
        cue.end_ms = end_ms
        pieces = _split_cue_pieces(cue, max_segment_ms)
        if len(pieces) <= 1:
            cue.index = next_index
            next_index += 1
            repaired.append(cue)
            last_end = cue.end_ms
            continue
        source_key = next(iter(cue.texts), "source")
        for offset, (piece_start, piece_end, piece) in enumerate(pieces):
            child = Cue(
                cue_id=f"{cue.cue_id}-p{offset + 1}",
                index=next_index,
                start_ms=piece_start,
                end_ms=piece_end if piece_end > piece_start else min(cue.end_ms, piece_start + 200),
                texts={source_key: piece},
                kind=cue.kind,
                issues=list(cue.issues) + [Issue(code="asr_split", message="按说话停顿或句号切开", cue_id=cue.cue_id)],
            )
            next_index += 1
            repaired.append(child)
        last_end = repaired[-1].end_ms
    graph.cues = repaired
    graph.duration_ms = max((item.end_ms for item in repaired), default=graph.duration_ms)
    return graph
