"""特效顶注。不做卡拉 OK 和飞字。

人物 / 专名第一次出现写一句 8–16 字的 \\an8 顶注。
检索失败标未证实，不编造空话。写不下的进 Briefing。
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional

from ..core.logging import studio_log
from ..core.models import Cue, CueGraph, Issue

CJK_NAME = re.compile(r"[\u3400-\u9fff]{2,8}")
LATIN_NAME = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})\b")


def extract_terms(text: str) -> List[str]:
    found = []
    for match in list(CJK_NAME.findall(text or "")) + list(LATIN_NAME.findall(text or "")):
        item = match.strip()
        if item and item not in found and len(item) >= 2:
            found.append(item)
    return found


def clip_brief(text: str) -> str:
    body = re.sub(r"\s+", "", str(text or ""))
    if len(body) <= 16:
        return body
    return body[:16]


class EffectsBuilder:
    def __init__(self, config: Dict[str, Any], *, research: Optional[Callable[..., str]] = None, logger: Any = None):
        self.config = config
        self.research = research
        self.logger = logger

    def apply(self, graph: CueGraph, *, title: str = "") -> CueGraph:
        if not self.config.get("effects_enabled"):
            graph.notes = []
            graph.briefing = []
            return graph
        density_ms = int(float(self.config.get("effects_density_minutes") or 2.5) * 60 * 1000)
        seen = set()
        last_note_at = -10**9
        notes: List[Cue] = []
        briefing: List[Dict[str, Any]] = []
        if self.config.get("effects_character_briefs", True) or self.config.get("effects_keyword_notes", True):
            for cue in graph.sorted_cues():
                terms = extract_terms(cue.text())
                for term in terms:
                    if term in seen:
                        continue
                    seen.add(term)
                    if cue.start_ms - last_note_at < density_ms and notes:
                        if self.config.get("effects_briefing", True):
                            briefing.append({"term": term, "cue_id": cue.cue_id, "text": f"{term}（密度限制，见 Briefing）"})
                        continue
                    brief, confirmed = self._research_term(term, title)
                    if not brief:
                        cue.issues.append(Issue(code="unverified_note", message=f"{term} 未证实", cue_id=cue.cue_id))
                        continue
                    short = clip_brief(brief)
                    if len(brief) > 16 and self.config.get("effects_briefing", True):
                        briefing.append({"term": term, "cue_id": cue.cue_id, "text": brief, "confirmed": confirmed})
                    note = Cue(
                        cue_id=f"note-{cue.cue_id}-{len(notes)+1}",
                        index=len(notes) + 1,
                        start_ms=cue.start_ms,
                        end_ms=min(cue.end_ms, cue.start_ms + 4000),
                        texts={"note": short},
                        kind="note",
                    )
                    if not confirmed:
                        note.issues.append(Issue(code="unverified_note", message="检索未证实", cue_id=note.cue_id))
                    notes.append(note)
                    last_note_at = cue.start_ms
                    break
        graph.notes = notes
        graph.briefing = briefing
        return graph

    def _research_term(self, term: str, title: str) -> tuple:
        if not self.research:
            # 没有检索线路时不编圆，只给未证实占位。
            return "", False
        try:
            raw = self.research(
                f"用不超过 16 个汉字解释影视《{title}》里的「{term}」。只写已出现信息，不要剧透后文。不要加引号。",
                system="你是字幕组资料员。只输出一句短注。不确定就输出 UNVERIFIED。",
                role="research",
            )
        except Exception as exc:  # noqa: BLE001
            studio_log(self.logger, "warning", "检索失败 %s：%s", term, exc)
            return "", False
        text = str(raw or "").strip()
        if not text or "UNVERIFIED" in text.upper():
            return "", False
        return text, True
