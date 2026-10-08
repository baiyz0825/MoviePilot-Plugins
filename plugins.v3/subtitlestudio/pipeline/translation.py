"""对白翻译：免费引擎 + 大模型，批量必须带上下文，脏 JSON 先修格式。

现网批量路径丢掉 context_window；这里每批都带前后句。
失败默认保留原文并标 Issue，不写「[翻译失败]」。
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..core.format_repair import repair_model_batch
from ..core.logging import studio_log
from ..core.models import CueGraph, Issue
from .mt_engines import FreeMtRouter

LlmFn = Callable[..., str]


TRANSLATE_SYSTEM = (
    "你是字幕翻译器。只输出 JSON 数组，不要解释。"
    "数组元素格式：{\"id\":\"<cue_id>\",\"zh\":\"译文\"}。"
    "id 必须来自输入，不要改编号。不要把注释写进对白。"
)


def build_batch_prompt(items: Sequence[Dict[str, str]], context_window: int) -> str:
    lines = ["请翻译下列字幕。必须按 id 返回 JSON 数组。上下文仅供理解，不要翻译上下文。", ""]
    for index, item in enumerate(items):
        before = items[max(0, index - context_window):index]
        after = items[index + 1:index + 1 + context_window]
        if before:
            lines.append("上文：" + " / ".join(part["text"] for part in before))
        lines.append(f"[{item['id']}] {item['text']}")
        if after:
            lines.append("下文：" + " / ".join(part["text"] for part in after))
        lines.append("")
    return "\n".join(lines)


class TranslationService:
    def __init__(
        self,
        config: Dict[str, Any],
        *,
        http: Callable[..., Any],
        llm: Optional[LlmFn] = None,
        logger: Any = None,
    ):
        self.config = config
        self.http = http
        self.llm = llm
        self.logger = logger
        self.mt = FreeMtRouter(http, config, logger)

    def translate_graph(self, graph: CueGraph, target_langs: Sequence[str]) -> CueGraph:
        if not self.config.get("translate_enabled", True):
            return graph
        targets = [item for item in target_langs if item and item != graph.source_lang] or ["zh-Hans"]
        for lang in targets:
            self._translate_lang(graph, lang)
        return graph

    def _translate_lang(self, graph: CueGraph, lang: str) -> None:
        pending = [cue for cue in graph.sorted_cues() if not cue.text(lang)]
        if not pending:
            studio_log(self.logger, "info", "语种 %s 已有译文，跳过", lang)
            return
        backend = str(self.config.get("translate_backend") or "free_first")
        studio_log(self.logger, "info", "翻译语种 %s 待译 %s 句 backend=%s", lang, len(pending), backend)
        if backend in {"free_first", "free_only"}:
            try:
                texts = [cue.text() for cue in pending]
                translated = self.mt.translate_batch(texts, source=graph.source_lang, target=lang)
                for cue, text in zip(pending, translated):
                    if text:
                        cue.texts[lang] = text
            except Exception as exc:  # noqa: BLE001
                if backend == "free_only":
                    self._mark_failed(pending, lang, str(exc))
                    return
                studio_log(self.logger, "warning", "免费引擎失败，改走大模型：%s", exc)
        leftover = [cue for cue in pending if not cue.text(lang)]
        filled = len(pending) - len(leftover)
        if filled:
            studio_log(self.logger, "info", "免费引擎译出 %s/%s 句 lang=%s", filled, len(pending), lang)
        if leftover and backend != "free_only":
            studio_log(self.logger, "info", "大模型补译 %s 句 lang=%s", len(leftover), lang)
            self._translate_with_llm(leftover, lang)

    def _translate_with_llm(self, cues, lang: str) -> None:
        if not self.llm:
            self._mark_failed(cues, lang, "没有可用的大模型线路")
            return
        batch_size = int(self.config.get("batch_size") or 20)
        context_window = int(self.config.get("context_window") or 5)
        enable_batch = bool(self.config.get("enable_batch", True))
        partial = bool(self.config.get("repair_partial_accept", True))
        items = [{"id": cue.cue_id, "text": cue.text()} for cue in cues]
        chunks = [items[i:i + batch_size] for i in range(0, len(items), batch_size)] if enable_batch else [[item] for item in items]
        lookup = {cue.cue_id: cue for cue in cues}
        total = len(chunks)
        for index, chunk in enumerate(chunks, start=1):
            if total > 1 and (index == 1 or index == total or index % max(1, total // 10) == 0):
                studio_log(self.logger, "info", "大模型翻译进度 %s/%s 批 lang=%s", index, total, lang)
            self._run_chunk(chunk, lookup, lang, context_window, partial)

    def _run_chunk(self, chunk, lookup, lang: str, context_window: int, partial: bool) -> None:
        expected = [item["id"] for item in chunk]
        prompt = build_batch_prompt(chunk, context_window)
        raw = self.llm(prompt, system=TRANSLATE_SYSTEM, role="translate")
        mapped, missing, _steps = repair_model_batch(raw, expected, partial_accept=partial) if self.config.get("format_repair_enabled", True) else ({}, expected, [])
        if not mapped and not self.config.get("format_repair_enabled", True):
            try:
                payload = json.loads(raw)
                mapped = {str(item.get("id")): str(item.get("zh") or item.get("text") or "") for item in payload}
                missing = [item for item in expected if not mapped.get(item)]
            except Exception:  # noqa: BLE001
                mapped, missing = {}, expected
        for cue_id, text in mapped.items():
            if cue_id in lookup and text:
                lookup[cue_id].texts[lang] = text
        if missing and len(chunk) > 1:
            half = max(1, len(chunk) // 2)
            self._run_chunk(chunk[:half], lookup, lang, context_window, partial)
            self._run_chunk(chunk[half:], lookup, lang, context_window, partial)
            return
        if missing and len(chunk) == 1:
            cue = lookup[missing[0]]
            if self.config.get("write_failure_placeholder"):
                cue.texts[lang] = "[翻译失败]"
            cue.issues.append(Issue(code="translate_failed", message="该句未能对齐译文", cue_id=cue.cue_id))

    def _mark_failed(self, cues, lang: str, message: str) -> None:
        for cue in cues:
            if self.config.get("write_failure_placeholder"):
                cue.texts[lang] = "[翻译失败]"
            cue.issues.append(Issue(code="translate_failed", message=message, cue_id=cue.cue_id))

    def failure_rate(self, graph: CueGraph, lang: str) -> float:
        cues = graph.sorted_cues()
        if not cues:
            return 0.0
        failed = sum(1 for cue in cues if any(issue.code == "translate_failed" for issue in cue.issues) and not cue.text(lang))
        return failed / len(cues)
