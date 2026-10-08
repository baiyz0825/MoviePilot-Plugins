"""免费机器翻译。对白可以走这些引擎；质检 / 纠错 / 人物检索不走这里。

HTTP 由宿主注入，领域层不写 httpx / httpx2，避免 V2 / V3 串代。
"""

from __future__ import annotations

import json
import re
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote

HttpFn = Callable[..., Any]


def _edge_lang(lang: str) -> str:
    mapping = {
        "zh-Hans": "zh-Hans",
        "zh-Hant": "zh-Hant",
        "en": "en",
        "ja": "ja",
        "ko": "ko",
        "source": "en",
    }
    return mapping.get(lang, lang)


class FreeMtRouter:
    def __init__(self, http: HttpFn, config: Dict[str, Any], logger: Any = None):
        self.http = http
        self.config = config
        self.logger = logger

    def translate_batch(self, texts: List[str], *, source: str, target: str) -> List[Optional[str]]:
        engines = list(self.config.get("mt_engines") or ["edge", "gtx"])
        last_error = ""
        for engine in engines:
            try:
                if engine == "edge":
                    return self._edge(texts, source, target)
                if engine == "gtx":
                    return self._gtx(texts, source, target)
                if engine == "deeplx":
                    return self._deeplx(texts, source, target)
                if engine == "libre":
                    return self._libre(texts, source, target)
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                if self.logger:
                    self.logger.warning("[SubtitleStudio] 免费引擎 %s 失败：%s", engine, exc)
        raise RuntimeError(last_error or "免费翻译引擎全部失败")

    def _get(self, url: str, **kwargs: Any) -> Any:
        return self.http("GET", url, **kwargs)

    def _post(self, url: str, **kwargs: Any) -> Any:
        return self.http("POST", url, **kwargs)

    def _edge(self, texts: List[str], source: str, target: str) -> List[Optional[str]]:
        # Edge 公开接口不需要 Key，适合作为默认第一档。
        url = "https://api.cognitive.microsofttranslator.com/translate"
        headers = {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": "",
        }
        # 无 Key 时走网页翻译端点的简化实现：逐句调 gtx 风格的 Edge 兼容地址。
        results = []
        for text in texts:
            query = (
                "https://edge.microsoft.com/translate/translatetext"
                f"?from={quote(_edge_lang(source))}&to={quote(_edge_lang(target))}"
            )
            payload = self._post(query, json_body=[{"Text": text}], headers={"Content-Type": "application/json"})
            results.append(_pick_translated(payload, text))
        return results

    def _gtx(self, texts: List[str], source: str, target: str) -> List[Optional[str]]:
        results = []
        for text in texts:
            url = (
                "https://translate.googleapis.com/translate_a/single"
                f"?client=gtx&sl={quote(source or 'auto')}&tl={quote(_edge_lang(target))}&dt=t&q={quote(text)}"
            )
            payload = self._get(url)
            results.append(_pick_gtx(payload, text))
        return results

    def _deeplx(self, texts: List[str], source: str, target: str) -> List[Optional[str]]:
        url = str(self.config.get("deeplx_url") or "").rstrip("/")
        if not url:
            raise RuntimeError("未配置 DeepLX 地址")
        results = []
        for text in texts:
            payload = self._post(
                f"{url}/translate",
                json_body={"text": text, "source_lang": source, "target_lang": target},
            )
            if isinstance(payload, dict):
                results.append(str(payload.get("data") or payload.get("text") or "") or None)
            else:
                results.append(None)
        return results

    def _libre(self, texts: List[str], source: str, target: str) -> List[Optional[str]]:
        url = str(self.config.get("libre_url") or "").rstrip("/")
        if not url:
            raise RuntimeError("未配置 LibreTranslate 地址")
        results = []
        for text in texts:
            payload = self._post(
                f"{url}/translate",
                json_body={"q": text, "source": source or "auto", "target": target, "format": "text"},
            )
            if isinstance(payload, dict):
                results.append(str(payload.get("translatedText") or "") or None)
            else:
                results.append(None)
        return results


def _pick_translated(payload: Any, original: str) -> Optional[str]:
    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict):
            translations = first.get("translations") or []
            if translations and isinstance(translations[0], dict):
                return str(translations[0].get("text") or "") or None
        if isinstance(first, str):
            return first
    if isinstance(payload, dict):
        return str(payload.get("text") or payload.get("translation") or "") or None
    if isinstance(payload, str) and payload.strip() and payload.strip() != original:
        return payload.strip()
    return None


def _pick_gtx(payload: Any, original: str) -> Optional[str]:
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            return payload if payload != original else None
    if isinstance(payload, list) and payload and isinstance(payload[0], list):
        chunks = [item[0] for item in payload[0] if isinstance(item, list) and item]
        text = "".join(str(item) for item in chunks if item)
        return text or None
    return None
