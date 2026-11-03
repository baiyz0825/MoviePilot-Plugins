"""在线字幕源。HTTP 由宿主注入。

ASSRT / OpenSubtitles 走官方 JSON。SubHD / Zimuku 只做可测试的结果解析，
真正的验证码弹窗由页面 inject('moviepilot:dialog') 承接，不在领域层弹窗。
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List
from urllib.parse import quote

from ..core.errors import looks_like_subtitle
from ..core.logging import studio_log
from .local import list_sidecars


RESULT_HREF = re.compile(r'href="([^"]+)"[^>]*>([^<]+)')


class OnlineSearchService:
    def __init__(self, http: Callable[..., Any], config: Dict[str, Any], logger: Any = None):
        self.http = http
        self.config = config
        self.logger = logger

    def search(self, job, config: Dict[str, Any] | None = None) -> List[Dict[str, Any]]:
        cfg = config or self.config
        providers = list(cfg.get("online_providers") or [])
        studio_log(self.logger, "info", "在线搜索 title=%s providers=%s", job.title, ",".join(providers) or "无")
        results: List[Dict[str, Any]] = []
        for provider in providers:
            try:
                if provider == "assrt" and cfg.get("assrt_api_key"):
                    results.extend(self._assrt(job.title, cfg))
                elif provider == "opensubtitles" and cfg.get("opensubtitles_api_key"):
                    results.extend(self._opensubtitles(job, cfg))
                elif provider == "subhd":
                    results.extend(self._html_search(cfg.get("subhd_url") or "https://subhd.tv", job.title, "subhd"))
                elif provider == "zimuku":
                    results.extend(self._html_search(cfg.get("zimuku_url") or "https://zmk.pw", job.title, "zimuku"))
            except Exception as exc:  # noqa: BLE001
                studio_log(self.logger, "warning", "搜索 %s 失败：%s", provider, exc)
            else:
                studio_log(self.logger, "info", "搜索 %s 得到 %s 条", provider, len([item for item in results if item.get("provider") == provider]))
        results = self._rank(results, cfg)
        studio_log(self.logger, "info", "在线搜索合计 %s 条", len(results))
        return results

    def best_download(self, job, config: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
        cfg = config or self.config
        local = list_sidecars(job.path)
        if local:
            path = PathLike(local[0]["path"])
            content = path.read_text(encoding="utf-8", errors="ignore")
            if looks_like_subtitle(content):
                studio_log(self.logger, "info", "搜索前已有本地外挂 %s", local[0]["filename"])
                return {"filename": local[0]["filename"], "content": content, "lang": "source"}
        results = self.search(job, cfg)
        if not results:
            studio_log(self.logger, "info", "在线搜索无结果 title=%s", job.title)
            return None
        for index, top in enumerate(results[:12]):
            studio_log(self.logger, "info", "尝试在线结果 #%s %s / %s", index + 1, top.get("provider"), top.get("title") or top.get("filename"))
            content = top.get("content")
            if not content:
                url = top.get("download_url") or top.get("url")
                if not url:
                    continue
                try:
                    payload = self.http("GET", url)
                except Exception as exc:  # noqa: BLE001
                    studio_log(self.logger, "warning", "下载字幕失败：%s", exc)
                    continue
                if isinstance(payload, bytes):
                    content = payload.decode("utf-8", errors="ignore")
                elif isinstance(payload, dict):
                    # OpenSubtitles 等可能返回 JSON 包装
                    content = str(payload.get("content") or payload.get("data") or "")
                else:
                    content = str(payload or "")
            if looks_like_subtitle(content):
                return {**top, "content": content}
            studio_log(self.logger, "info", "结果正文不像字幕，跳过 provider=%s", top.get("provider"))
        studio_log(self.logger, "info", "在线结果均无可用正文 title=%s", job.title)
        return None

    def _rank(self, results: List[Dict[str, Any]], cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        lang_order = list(cfg.get("search_language_priority") or ["bilingual", "zh-Hans", "zh-Hant", "en"])
        fmt_order = list(cfg.get("search_format_priority") or ["ass", "srt", "ssa", "vtt"])

        def score(item: Dict[str, Any]) -> tuple:
            title = (item.get("title") or item.get("filename") or "").lower()
            lang = str(item.get("lang") or "")
            lang_rank = lang_order.index(lang) if lang in lang_order else 50
            if "双语" in title or "bilingual" in title:
                lang_rank = min(lang_rank, lang_order.index("bilingual") if "bilingual" in lang_order else 0)
            fmt_rank = 50
            for index, fmt in enumerate(fmt_order):
                if title.endswith(f".{fmt}") or f".{fmt}" in title or item.get("format") == fmt:
                    fmt_rank = index
                    break
            effects_boost = 0
            if cfg.get("effects_enabled") and re.search(r"特效|解说", item.get("title") or ""):
                effects_boost = -1
            return (effects_boost, lang_rank, fmt_rank, item.get("title") or "")

        return sorted(results, key=score)

    def _assrt(self, title: str, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        token = quote(cfg.get("assrt_api_key") or "")
        base = (cfg.get("assrt_api_url") or "https://api.assrt.net").rstrip("/")
        # token 仍走 query（ASSRT 合同），调用方异常文案会经 user_message 消毒
        url = f"{base}/v1/sub/search?q={quote(title)}&token={token}"
        payload = self.http("GET", url)
        rows = []
        items = payload.get("sub", {}).get("subs") if isinstance(payload, dict) else []
        for item in items or []:
            rows.append({
                "provider": "assrt",
                "title": item.get("native_name") or item.get("videoname") or title,
                "url": item.get("url") or "",
                "download_url": item.get("url") or "",
                "lang": "zh-Hans",
                "format": "ass" if str(item.get("native_name") or "").lower().endswith(".ass") else "",
            })
        return rows

    def _opensubtitles(self, job, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        url = f"{cfg.get('opensubtitles_api_url') or 'https://api.opensubtitles.com/api/v1'}/subtitles?query={quote(job.title)}"
        payload = self.http("GET", url, headers={"Api-Key": cfg.get("opensubtitles_api_key") or ""})
        rows = []
        for item in (payload.get("data") if isinstance(payload, dict) else []) or []:
            attrs = item.get("attributes") or {}
            files = attrs.get("files") or []
            file_id = files[0].get("file_id") if files and isinstance(files[0], dict) else ""
            download_url = ""
            if file_id:
                # 真实下载需另调 /download；此处保留占位，best_download 会因无正文跳过并试下一条
                download_url = f"{cfg.get('opensubtitles_api_url') or 'https://api.opensubtitles.com/api/v1'}/download"
            rows.append({
                "provider": "opensubtitles",
                "title": attrs.get("release") or job.title,
                "url": "",
                "download_url": download_url,
                "file_id": file_id,
                "lang": (attrs.get("language") or "en"),
            })
        return rows

    def _html_search(self, root: str, title: str, provider: str) -> List[Dict[str, Any]]:
        html = self.http("GET", f"{root.rstrip('/')}/search?q={quote(title)}")
        return parse_search_html(str(html or ""), provider, root)


def parse_search_html(html: str, provider: str, root: str) -> List[Dict[str, Any]]:
    """纯函数，方便单测喂夹具，不必打真实站点。"""
    rows = []
    for href, title in RESULT_HREF.findall(html or ""):
        if "search" in href and "q=" in href:
            continue
        if href.startswith("#"):
            continue
        url = href if href.startswith("http") else f"{root.rstrip('/')}/{href.lstrip('/')}"
        rows.append({"provider": provider, "title": title.strip(), "url": url, "download_url": url, "lang": "zh-Hans"})
        if len(rows) >= 20:
            break
    return rows


class PathLike:
    def __init__(self, path: str):
        self.path = path

    def read_text(self, encoding="utf-8", errors="ignore"):
        from pathlib import Path
        return Path(self.path).read_text(encoding=encoding, errors=errors)
