"""对白翻译：免费引擎 + 大模型，批量必须带上下文，脏 JSON 先修格式。

现网批量路径丢掉 context_window；这里每批都带前后句。
失败默认保留原文并标 Issue，不写「[翻译失败]」。
"""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List, Optional, Sequence

from ..core.format_repair import repair_model_batch, translation_keeps_cue
from ..core.language import (
    canonical_language,
    normalize_graph_source_language,
    translation_validation_reason,
)
from ..core.logging import studio_log
from ..core.models import CueGraph, Issue
from .mt_engines import FreeMtRouter

LlmFn = Callable[..., str]


TRANSLATE_SYSTEM = (
    "你是字幕翻译器。只输出 JSON 数组，不要解释。"
    "数组元素统一为：{\"id\":\"<cue_id>\",\"text\":\"translated text\"}。"
    "id 必须原样来自输入。只翻译当前 cue；上下文仅用于理解，禁止翻译或并入上下文。"
    "不要扩写、不要添加原文没有的内容，也不要返回原文与译文混合的段落。"
)


_LANGUAGE_NAMES = {
    "source": "自动识别的源语言",
    "zh-Hans": "简体中文 (zh-Hans)",
    "zh-Hant": "繁体中文 (zh-Hant)",
    "en": "英语 (en)",
    "ja": "日语 (ja)",
    "ko": "韩语 (ko)",
    "fr": "法语 (fr)",
    "de": "德语 (de)",
    "es": "西班牙语 (es)",
    "it": "意大利语 (it)",
    "pt": "葡萄牙语 (pt)",
    "ru": "俄语 (ru)",
    "ar": "阿拉伯语 (ar)",
    "hi": "印地语 (hi)",
    "bn": "孟加拉语 (bn)",
    "ur": "乌尔都语 (ur)",
    "id": "印度尼西亚语 (id)",
    "pcm": "尼日利亚皮钦语 (pcm)",
    "mr": "马拉地语 (mr)",
    "te": "泰卢固语 (te)",
    "tr": "土耳其语 (tr)",
    "ta": "泰米尔语 (ta)",
    "vi": "越南语 (vi)",
    "tl": "菲律宾语 (tl)",
    "fa": "波斯语 (fa)",
    "jv": "爪哇语 (jv)",
    "ha": "豪萨语 (ha)",
    "sw": "斯瓦希里语 (sw)",
}


def _language_name(lang: str) -> str:
    normalized = canonical_language(lang)
    return _LANGUAGE_NAMES.get(normalized, normalized)


def build_translate_system(source_lang: str, target_lang: str) -> str:
    source = canonical_language(source_lang)
    target = canonical_language(target_lang)
    return (
        f"{TRANSLATE_SYSTEM}本批源语言：{_language_name(source)}；"
        f"目标语言：{_language_name(target)}。所有 text 字段必须是目标语言。"
    )


def build_batch_prompt(
    items: Sequence[Dict[str, str]],
    context_window: int,
    source_lang: str = "source",
    target_lang: str = "zh-Hans",
) -> str:
    lines = [
        f"将当前字幕从 {_language_name(source_lang)} 翻译为 {_language_name(target_lang)}。",
        "必须保持每个 id，并按 {\"id\":\"...\",\"text\":\"...\"} 返回 JSON 数组。",
        "上文和下文只供理解；只翻译带 cue ID 的当前句，不得把上下文写入 text。",
        "",
    ]
    for index, item in enumerate(items):
        before = items[max(0, index - context_window):index]
        after = items[index + 1:index + 1 + context_window]
        if before:
            lines.append("上文（勿翻译）：" + " / ".join(part["text"] for part in before))
        lines.append(f"[{item['id']}] {item['text']}")
        if after:
            lines.append("下文（勿翻译）：" + " / ".join(part["text"] for part in after))
        lines.append("")
    return "\n".join(lines)


def _source_text(cue: Any, source_lang: str) -> str:
    return cue.text(source_lang) or cue.text("source") or cue.text()


class TranslationService:
    def __init__(
        self,
        config: Dict[str, Any],
        *,
        http: Callable[..., Any],
        llm: Optional[LlmFn] = None,
        logger: Any = None,
        cancel_check: Optional[Callable[[], None]] = None,
        on_progress: Optional[Callable[[float, str], None]] = None,
    ):
        self.config = config
        self.http = http
        self.llm = llm
        self.logger = logger
        self.cancel_check = cancel_check
        self.on_progress = on_progress
        self.mt = FreeMtRouter(http, config, logger, cancel_check=cancel_check)
        self._llm_calls = 0
        self._max_llm_calls = max(8, int(config.get("max_llm_calls_per_lang") or 120))
        self._max_repair_depth = max(1, int(config.get("max_repair_depth") or 4))
        self._done = 0
        self._total = 1

    def translate_graph(self, graph: CueGraph, target_langs: Sequence[str]) -> CueGraph:
        normalize_graph_source_language(graph)
        if not self.config.get("translate_enabled", True):
            return graph
        target_set = {canonical_language(item) for item in target_langs}
        targets = []
        for item in target_langs:
            lang = canonical_language(item)
            if lang != "source" and lang != graph.source_lang and lang not in targets:
                targets.append(lang)
        # Source text already satisfies a requested output language. The graph
        # normalizer has migrated it to that canonical key, so no alias copy is
        # created here (copying was the original duplicate-English bug).
        if graph.source_lang in target_set:
            for cue in graph.sorted_cues():
                source_text = _source_text(cue, graph.source_lang)
                if source_text:
                    cue.texts.setdefault(graph.source_lang, source_text)
        self._discard_invalid_existing_targets(graph, targets)
        if not targets:
            studio_log(self.logger, "info", "源语种 %s 已包含在目标语种中，跳过翻译", graph.source_lang)
            return graph
        pending_total = sum(
            1
            for lang in targets
            for cue in graph.sorted_cues()
            if not cue.text(lang)
        )
        self._total = max(1, pending_total)
        self._done = 0
        for lang in targets:
            self._translate_lang(graph, lang)
        return graph

    def _discard_invalid_existing_targets(self, graph: CueGraph, targets: Sequence[str]) -> None:
        for lang in targets:
            for cue in graph.sorted_cues():
                existing = cue.text(lang)
                source_text = _source_text(cue, graph.source_lang)
                if not existing or not source_text:
                    continue
                reason = translation_validation_reason(source_text, existing, lang, graph.source_lang)
                if not reason:
                    continue
                cue.texts.pop(lang, None)
                cue.issues.append(Issue(
                    code="invalid_translation",
                    message=f"已有 {lang} 文本不符合目标语言（{reason}），已重新进入翻译",
                    cue_id=cue.cue_id,
                ))

    def _tick(self, count: int, lang: str, *, force: bool = False) -> None:
        self._done = min(self._total, self._done + max(0, count))
        if not callable(self.on_progress):
            return
        if not force and count < 8 and self._done % 8 != 0 and self._done != self._total:
            return
        try:
            self.on_progress(self._done / self._total, f"翻译 {lang} · {self._done}/{self._total} 句")
        except Exception as exc:  # noqa: BLE001
            studio_log(self.logger, "warning", "翻译进度回报失败：%s", exc)

    def _translate_lang(self, graph: CueGraph, lang: str) -> None:
        if callable(self.cancel_check):
            self.cancel_check()
        pending = [cue for cue in graph.sorted_cues() if not cue.text(lang)]
        if not pending:
            studio_log(self.logger, "info", "语种 %s 已有译文，跳过", lang)
            return
        backend = str(self.config.get("translate_backend") or "free_first")
        studio_log(self.logger, "info", "翻译语种 %s 待译 %s 句 backend=%s", lang, len(pending), backend)
        self._tick(0, lang, force=True)
        if backend in {"free_first", "free_only"}:
            try:
                texts = [_source_text(cue, graph.source_lang) for cue in pending]
                translated = self.mt.translate_batch(
                    texts,
                    source=graph.source_lang,
                    target=lang,
                    on_item=lambda: self._tick(1, lang),
                )
                for cue, text in zip(pending, translated):
                    source_text = _source_text(cue, graph.source_lang)
                    reason = translation_validation_reason(source_text, text or "", lang, graph.source_lang)
                    if text and not reason:
                        cue.texts[lang] = text
            except Exception as exc:  # noqa: BLE001
                if backend == "free_only":
                    self._mark_failed(pending, lang, "免费翻译失败")
                    return
                studio_log(self.logger, "warning", "免费引擎失败，改走大模型：%s", exc)
        leftover = [cue for cue in pending if not cue.text(lang)]
        filled = len(pending) - len(leftover)
        if filled:
            studio_log(self.logger, "info", "免费引擎译出 %s/%s 句 lang=%s", filled, len(pending), lang)
        if leftover and backend == "free_only":
            self._mark_failed(leftover, lang, "免费翻译未返回有效目标语言文本")
        elif leftover:
            studio_log(self.logger, "info", "大模型补译 %s 句 lang=%s", len(leftover), lang)
            self._translate_with_llm(leftover, lang, graph.source_lang)

    def _translate_with_llm(self, cues, lang: str, source_lang: str) -> None:
        if not self.llm:
            self._mark_failed(cues, lang, "没有可用的大模型线路")
            return
        batch_size = int(self.config.get("batch_size") or 20)
        context_window = int(self.config.get("context_window") or 5)
        enable_batch = bool(self.config.get("enable_batch", True))
        partial = bool(self.config.get("repair_partial_accept", True))
        items = [{"id": cue.cue_id, "text": _source_text(cue, source_lang)} for cue in cues]
        chunks = [items[i:i + batch_size] for i in range(0, len(items), batch_size)] if enable_batch else [[item] for item in items]
        lookup = {cue.cue_id: cue for cue in cues}
        total = len(chunks)
        for index, chunk in enumerate(chunks, start=1):
            if callable(self.cancel_check):
                self.cancel_check()
            if self._llm_calls >= self._max_llm_calls:
                studio_log(self.logger, "warning", "大模型调用预算用尽 lang=%s", lang)
                self._mark_failed([lookup[item["id"]] for item in chunk if item["id"] in lookup], lang, "大模型调用预算用尽")
                continue
            if total > 1 and (index == 1 or index == total or index % max(1, total // 10) == 0):
                studio_log(self.logger, "info", "大模型翻译进度 %s/%s 批 lang=%s", index, total, lang)
            self._run_chunk(chunk, lookup, lang, source_lang, context_window, partial, depth=0)
            if str(self.config.get("translate_backend") or "") not in {"free_first", "free_only"}:
                self._tick(len(chunk), lang, force=True)
            elif callable(self.on_progress):
                try:
                    self.on_progress(self._done / self._total, f"大模型补译 {lang} · {index}/{total} 批")
                except Exception as exc:  # noqa: BLE001
                    studio_log(self.logger, "warning", "翻译进度回报失败：%s", exc)

    def _run_chunk(self, chunk, lookup, lang: str, source_lang: str, context_window: int, partial: bool, *, depth: int) -> None:
        if callable(self.cancel_check):
            self.cancel_check()
        if self._llm_calls >= self._max_llm_calls:
            self._mark_failed([lookup[item["id"]] for item in chunk if item["id"] in lookup], lang, "大模型调用预算用尽")
            return
        expected = [item["id"] for item in chunk]
        prompt = build_batch_prompt(chunk, context_window, source_lang, lang)
        self._llm_calls += 1
        raw = self.llm(prompt, system=build_translate_system(source_lang, lang), role="translate")
        mapped, missing, _steps = repair_model_batch(raw, expected, partial_accept=partial) if self.config.get("format_repair_enabled", True) else ({}, expected, [])
        if not mapped and not self.config.get("format_repair_enabled", True):
            try:
                payload = json.loads(raw)
                mapped = {
                    str(item.get("id")): str(
                        item.get("text") or item.get("translation") or item.get("target") or item.get("zh") or item.get("en") or ""
                    )
                    for item in payload
                }
                missing = [item for item in expected if not mapped.get(item)]
            except Exception:  # noqa: BLE001
                mapped, missing = {}, expected
        rejected = []
        rejection_reasons = {}
        for cue_id, text in mapped.items():
            cue = lookup.get(cue_id)
            if not cue or not text:
                continue
            source_text = _source_text(cue, source_lang)
            reason = translation_validation_reason(source_text, text, lang, source_lang)
            if reason:
                rejected.append(cue_id)
                rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1
                continue
            if not translation_keeps_cue(source_text, text):
                rejected.append(cue_id)
                rejection_reasons["context_merged"] = rejection_reasons.get("context_merged", 0) + 1
                continue
            cue.texts[lang] = text
        if rejected:
            studio_log(self.logger, "info", "大模型译文校验未通过，退回重试 %s 句 reasons=%s", len(rejected), rejection_reasons)
            missing = list(dict.fromkeys([*missing, *rejected]))
        if missing and len(chunk) > 1 and depth < self._max_repair_depth:
            half = max(1, len(chunk) // 2)
            self._run_chunk(chunk[:half], lookup, lang, source_lang, context_window, partial, depth=depth + 1)
            self._run_chunk(chunk[half:], lookup, lang, source_lang, context_window, partial, depth=depth + 1)
            return
        if missing:
            for cue_id in missing:
                cue = lookup.get(cue_id)
                if not cue:
                    continue
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
