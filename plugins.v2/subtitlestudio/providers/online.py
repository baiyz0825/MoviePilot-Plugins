"""在线字幕源。HTTP 由宿主注入。

ASSRT / OpenSubtitles 走官方 JSON。SubHD / Zimuku 只做可测试的结果解析，
真正的验证码弹窗由页面 inject('moviepilot:dialog') 承接，不在领域层弹窗。
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List
from urllib.parse import quote

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
        if cfg.get("effects_enabled"):
            results.sort(key=lambda item: (0 if re.search(r"特效|解说", item.get("title") or "") else 1, item.get("title") or ""))
        studio_log(self.logger, "info", "在线搜索合计 %s 条", len(results))
        return results

    def best_download(self, job, config: Dict[str, Any] | None = None) -> Dict[str, Any] | None:
        cfg = config or self.config
        local = list_sidecars(job.path)
        if local:
            path = PathLike(local[0]["path"])
            studio_log(self.logger, "info", "搜索前已有本地外挂 %s", local[0]["filename"])
            return {"filename": local[0]["filename"], "content": path.read_text(encoding="utf-8", errors="ignore"), "lang": "source"}
        results = self.search(job, cfg)
        if not results:
            studio_log(self.logger, "info", "在线搜索无结果 title=%s", job.title)
            return None
        top = results[0]
        studio_log(self.logger, "info", "选用 %s / %s", top.get("provider"), top.get("title") or top.get("filename"))
        if top.get("content"):
            return top
        url = top.get("download_url") or top.get("url")
        if not url:
            return top
        payload = self.http("GET", url)
        if isinstance(payload, bytes):
            content = payload.decode("utf-8", errors="ignore")
        else:
            content = str(payload or "")
        return {**top, "content": content}

    def _assrt(self, title: str, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        url = f"{cfg.get('assrt_api_url') or 'https://api.assrt.net'}/v1/sub/search?q={quote(title)}&token={quote(cfg.get('assrt_api_key') or '')}"
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
            })
        return rows

    def _opensubtitles(self, job, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
        url = f"{cfg.get('opensubtitles_api_url') or 'https://api.opensubtitles.com/api/v1'}/subtitles?query={quote(job.title)}"
        payload = self.http("GET", url, headers={"Api-Key": cfg.get("opensubtitles_api_key") or ""})
        rows = []
        for item in (payload.get("data") if isinstance(payload, dict) else []) or []:
            attrs = item.get("attributes") or {}
            rows.append({
                "provider": "opensubtitles",
                "title": attrs.get("release") or job.title,
                "url": "",
                "download_url": "",
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
