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


def chinese_cps_limit(duration_ms: int) -> int:
    # 中文阅读大约 9–11 字/秒，这里取 10，且最多两行约 32 字。
    seconds = max(0.4, duration_ms / 1000)
    return max(8, min(32, int(seconds * 10)))


def english_cps_limit(duration_ms: int) -> int:
    seconds = max(0.4, duration_ms / 1000)
    return max(12, min(84, int(seconds * 17)))


def split_by_cps(text: str, duration_ms: int, lang: str) -> List[str]:
    limit = chinese_cps_limit(duration_ms) if _looks_cjk(text or lang) else english_cps_limit(duration_ms)
    body = str(text or "").strip()
    if len(body) <= limit:
        return [body] if body else []
    chunks = []
    remaining = body
    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break
        cut = remaining.rfind("，", 0, limit) if _looks_cjk(remaining) else remaining.rfind(" ", 0, limit)
        if cut < max(4, limit // 3):
            cut = limit
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip(" ，,")
        if len(chunks) >= 2:
            if remaining:
                chunks[-1] = (chunks[-1] + (" " if not _looks_cjk(chunks[-1]) else "") + remaining).strip()
            break
    return [item for item in chunks if item]


def _looks_cjk(text: str) -> bool:
    return bool(re.search(r"[\u3400-\u9fff]", text or ""))


def repair_asr_graph(graph: CueGraph, *, enabled: bool = True) -> CueGraph:
    """丢掉空句/幻觉，重叠裁开，倒序对调，过长按 CPS 切开。"""
    if not enabled:
        return graph
    repaired: List[Cue] = []
    last_end = 0
    next_index = 1
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
        pieces = split_by_cps(text, cue.duration_ms(), cue.text())
        if len(pieces) <= 1:
            cue.index = next_index
            next_index += 1
            repaired.append(cue)
            last_end = cue.end_ms
            continue
        span = max(400, cue.duration_ms() // len(pieces))
        for offset, piece in enumerate(pieces):
            child = Cue(
                cue_id=f"{cue.cue_id}-p{offset + 1}",
                index=next_index,
                start_ms=cue.start_ms + offset * span,
                end_ms=cue.start_ms + (offset + 1) * span if offset + 1 < len(pieces) else cue.end_ms,
                texts={key: piece if key == next(iter(cue.texts), "source") else piece for key in (cue.texts or {"source": piece})},
                kind=cue.kind,
                issues=list(cue.issues) + [Issue(code="asr_split", message="过长句已按 CPS 切开", cue_id=cue.cue_id)],
            )
            next_index += 1
            repaired.append(child)
        last_end = repaired[-1].end_ms
    graph.cues = repaired
    graph.duration_ms = max((item.end_ms for item in repaired), default=graph.duration_ms)
    return graph
