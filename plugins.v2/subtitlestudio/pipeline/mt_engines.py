"""免费机器翻译。对白可以走这些引擎；质检 / 纠错 / 人物检索不走这里。

HTTP 由宿主注入，领域层不写 httpx / httpx2，避免 V2 / V3 串代。
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote

from ..core.language import can_remain_unchanged, translation_validation_reason
from ..core.logging import studio_log

HttpFn = Callable[..., Any]


def _edge_lang(lang: str) -> str:
    mapping = {
        "zh-Hans": "zh-Hans",
        "zh-Hant": "zh-Hant",
        "en": "en",
        "ja": "ja",
        "ko": "ko",
        "source": "auto",
    }
    return mapping.get(lang, lang)


class FreeMtRouter:
    def __init__(self, http: HttpFn, config: Dict[str, Any], logger: Any = None, cancel_check=None):
        self.http = http
        self.config = config
        self.logger = logger
        self.cancel_check = cancel_check
        self._max_consecutive_errors = max(2, int(config.get("mt_max_consecutive_errors") or 5))

    def translate_batch(self, texts: List[str], *, source: str, target: str, on_item=None) -> List[Optional[str]]:
        """Try later engines for each missing or invalid result, not just whole-batch errors."""
        engines = list(self.config.get("mt_engines") or ["edge", "gtx"])
        results: List[Optional[str]] = [None] * len(texts)
        pending = list(range(len(texts)))
        for engine in engines:
            if not pending:
                break
            if callable(self.cancel_check):
                self.cancel_check()
            functions = {
                "edge": lambda text: self._edge_one(text, source, target),
                "gtx": lambda text: self._gtx_one(text, source, target),
                "deeplx": lambda text: self._deeplx_one(text, source, target),
                "libre": lambda text: self._libre_one(text, source, target),
            }
            fn = functions.get(engine)
            if fn is None:
                studio_log(self.logger, "warning", "未知免费翻译引擎，已跳过：%s", engine)
                continue
            current = list(pending)
            try:
                translated = self._map_texts([texts[index] for index in current], fn, engine)
            except Exception as exc:  # noqa: BLE001
                studio_log(self.logger, "warning", "免费引擎 %s 失败：%s", engine, exc)
                continue
            for index, candidate in zip(current, translated):
                reason = translation_validation_reason(texts[index], candidate or "", target, source)
                if reason:
                    studio_log(self.logger, "debug", "%s 译文无效，尝试下一线路 index=%s reason=%s", engine, index, reason)
                    continue
                results[index] = str(candidate or "").strip()
            pending = [index for index in pending if not results[index]]
        if callable(on_item):
            for _item in texts:
                on_item()
        if pending:
            studio_log(self.logger, "warning", "免费翻译仍有 %s/%s 句无有效译文 target=%s", len(pending), len(texts), target)
        return results

    def _map_texts(self, texts: List[str], fn, engine: str, on_item=None) -> List[Optional[str]]:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        workers = max(1, min(8, int(self.config.get("parallel_workers") or 3)))
        results: List[Optional[str]] = [None] * len(texts)
        consecutive = 0
        # 小批量串行更稳；大批量才开线程池
        if len(texts) <= 4 or workers <= 1:
            for index, text in enumerate(texts):
                if callable(self.cancel_check):
                    self.cancel_check()
                try:
                    results[index] = fn(text)
                    consecutive = 0 if results[index] else consecutive + 1
                except Exception as exc:  # noqa: BLE001
                    consecutive += 1
                    studio_log(self.logger, "debug", "%s 单句失败：%s", engine, exc)
                if callable(on_item):
                    on_item()
                if consecutive >= self._max_consecutive_errors:
                    studio_log(self.logger, "debug", "%s 连续失败过多，保留已完成结果并切换线路", engine)
                    break
            return results
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(fn, text): index for index, text in enumerate(texts)}
            for future in as_completed(futures):
                if callable(self.cancel_check):
                    self.cancel_check()
                index = futures[future]
                try:
                    results[index] = future.result()
                    consecutive = 0 if results[index] else consecutive + 1
                except Exception as exc:  # noqa: BLE001
                    consecutive += 1
                    studio_log(self.logger, "debug", "%s 单句失败：%s", engine, exc)
                if callable(on_item):
                    on_item()
                if consecutive >= self._max_consecutive_errors * 2:
                    studio_log(self.logger, "debug", "%s 并发失败过多，保留已完成结果并切换线路", engine)
                    break
        return results

    def _get(self, url: str, **kwargs: Any) -> Any:
        return self.http("GET", url, **kwargs)

    def _post(self, url: str, **kwargs: Any) -> Any:
        return self.http("POST", url, **kwargs)

    def _edge_one(self, text: str, source: str, target: str) -> Optional[str]:
        query = (
            "https://edge.microsoft.com/translate/translatetext"
            f"?from={quote(_edge_lang(source))}&to={quote(_edge_lang(target))}"
        )
        payload = self._post(query, json_body=[{"Text": text}], headers={"Content-Type": "application/json"})
        return _pick_translated(payload, text)

    def _gtx_one(self, text: str, source: str, target: str) -> Optional[str]:
        url = (
            "https://translate.googleapis.com/translate_a/single"
            f"?client=gtx&sl={quote(_edge_lang(source or 'source'))}&tl={quote(_edge_lang(target))}&dt=t&q={quote(text)}"
        )
        payload = self._get(url)
        return _pick_gtx(payload, text)

    def _deeplx_one(self, text: str, source: str, target: str) -> Optional[str]:
        url = str(self.config.get("deeplx_url") or "").rstrip("/")
        if not url:
            raise RuntimeError("未配置 DeepLX 地址")
        payload = self._post(
            f"{url}/translate",
            json_body={"text": text, "source_lang": _edge_lang(source or "source"), "target_lang": target},
        )
        if isinstance(payload, dict):
            return str(payload.get("data") or payload.get("text") or "") or None
        return None

    def _libre_one(self, text: str, source: str, target: str) -> Optional[str]:
        url = str(self.config.get("libre_url") or "").rstrip("/")
        if not url:
            raise RuntimeError("未配置 LibreTranslate 地址")
        payload = self._post(
            f"{url}/translate",
            json_body={"q": text, "source": _edge_lang(source or "source"), "target": target, "format": "text"},
        )
        if isinstance(payload, dict):
            return str(payload.get("translatedText") or "") or None
        return None


def _different_candidate(value: Any, original: str) -> Optional[str]:
    candidate = str(value or "").strip()
    if not candidate:
        return None
    if re.sub(r"\s+", "", candidate).casefold() == re.sub(r"\s+", "", str(original or "")).casefold():
        return candidate if can_remain_unchanged(original) else None
    return candidate


def _pick_translated(payload: Any, original: str) -> Optional[str]:
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            translations = first.get("translations") or []
            if translations and isinstance(translations[0], dict):
                return _different_candidate(translations[0].get("text"), original)
        if isinstance(first, str):
            return _different_candidate(first, original)
    if isinstance(payload, dict):
        return _different_candidate(payload.get("text") or payload.get("translation"), original)
    if isinstance(payload, str):
        return _different_candidate(payload, original)
    return None


def _pick_gtx(payload: Any, original: str) -> Optional[str]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return _different_candidate(payload, original)
    if isinstance(payload, list) and payload and isinstance(payload[0], list):
        chunks = [item[0] for item in payload[0] if isinstance(item, list) and item]
        text = "".join(str(item) for item in chunks if item)
        return _different_candidate(text, original)
    return None
