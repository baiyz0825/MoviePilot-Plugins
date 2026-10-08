"""四个一级页共用的 API。

写操作和列表走同一 envelope。预览视频 / ASS 是原生流，不要包信封。
不要设匿名。页面调用 auth=bear。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..core.config_schema import FIELDS, PANES, apply_export_preset, default_config, normalize_plugin_config
from ..core.cuegraph import render_ass
from ..core.naming import extra_tracks, export_plan, sanitize_stem
from ..core.response import fail, ok
from ..ingest.gates import evaluate_gates


def build_api_routes(plugin) -> List[Dict[str, Any]]:
    return [
        _route("/status", plugin.api_status, ["GET"], "插件状态"),
        _route("/config", plugin.api_get_config, ["GET"], "读取配置"),
        _route("/config", plugin.api_save_config, ["POST"], "保存配置"),
        _route("/fields", plugin.api_fields, ["GET"], "设置页字段合同"),
        _route("/media", plugin.api_list_media, ["GET"], "媒体目录"),
        _route("/media/refresh", plugin.api_refresh_media, ["POST"], "刷新媒体目录"),
        _route("/jobs", plugin.api_list_jobs, ["GET"], "任务队列"),
        _route("/jobs", plugin.api_create_job, ["POST"], "入队"),
        _route("/jobs/{job_id}", plugin.api_get_job, ["GET"], "任务详情"),
        _route("/jobs/{job_id}/cut-in", plugin.api_cut_in, ["POST"], "插队"),
        _route("/jobs/{job_id}/priority", plugin.api_set_priority, ["POST"], "改优先级"),
        _route("/jobs/{job_id}/cancel", plugin.api_cancel_job, ["POST"], "取消任务"),
        _route("/jobs/{job_id}/retry", plugin.api_retry_job, ["POST"], "重跑任务"),
        _route("/jobs/{job_id}/cues", plugin.api_get_cues, ["GET"], "工作台 CueGraph"),
        _route("/jobs/{job_id}/cues/{cue_id}", plugin.api_save_cue, ["PUT"], "改一句"),
        _route("/jobs/{job_id}/export", plugin.api_export_job, ["POST"], "按当前图写盘"),
        _route("/jobs/{job_id}/search", plugin.api_search_job, ["POST"], "在线搜索"),
        _route("/endpoints/test", plugin.api_test_endpoint, ["POST"], "测连通"),
        _route("/endpoints/models", plugin.api_list_models, ["POST"], "拉模型列表"),
        {
            "path": "/jobs/{job_id}/preview/video",
            "endpoint": plugin.api_preview_video,
            "methods": ["GET"],
            "auth": "bear",
            "summary": "预览原片（Range）",
            "response_model": None,
        },
        {
            "path": "/jobs/{job_id}/preview/ass",
            "endpoint": plugin.api_preview_ass,
            "methods": ["GET"],
            "auth": "bear",
            "summary": "当前 CueGraph 打成预览 ASS",
            "response_model": None,
        },
    ]


def _route(path: str, endpoint, methods: List[str], summary: str) -> Dict[str, Any]:
    item = {
        "path": path,
        "endpoint": endpoint,
        "methods": methods,
        "auth": "bear",
        "summary": summary,
    }
    response_model = getattr(endpoint, "response_model", None)
    if response_model is not None:
        item["response_model"] = response_model
    return item


def export_preview(config: Dict[str, Any], path: str = "Movie.mkv", asr_ran: bool = False) -> List[Dict[str, Any]]:
    stem = sanitize_stem(path)
    return export_plan(config, stem) + extra_tracks(config, stem, asr_ran=asr_ran)


class StudioApiMixin:
    """挂在插件类上的端点实现。"""

    def api_status(self) -> Dict[str, Any]:
        counts = self.services.store.counts()
        return ok({
            "enabled": bool(self.get_state()),
            "generation": self.host_generation,
            "version": self.plugin_version,
            "counts": counts,
        })

    def api_get_config(self) -> Dict[str, Any]:
        return ok(normalize_plugin_config(self.get_config() or {}))

    def api_save_config(self, body: Dict[str, Any] = None) -> Dict[str, Any]:
        payload = normalize_plugin_config(body or {})
        if payload.get("export_preset"):
            payload = apply_export_preset(payload, payload["export_preset"])
        self.update_config(payload)
        self.init_plugin(payload)
        return ok(payload, "已保存")

    def api_fields(self) -> Dict[str, Any]:
        return ok({"fields": FIELDS, "panes": PANES, "defaults": default_config()})

    def api_list_media(self, q: str = "", media_type: str = "") -> Dict[str, Any]:
        return ok({"items": self.services.catalog.list_media(q=q, media_type=media_type)})

    def api_refresh_media(self) -> Dict[str, Any]:
        return self.api_list_media()

    def api_list_jobs(self, q: str = "", status: str = "") -> Dict[str, Any]:
        jobs = [item.to_dict() for item in self.services.store.list_jobs(q=q, status=status)]
        return ok({"items": jobs, "counts": self.services.store.counts()})

    def api_create_job(self, body: Dict[str, Any] = None) -> Dict[str, Any]:
        data = body or {}
        path = str(data.get("path") or "")
        config = normalize_plugin_config(self.get_config() or {})
        ok_gate, reason = evaluate_gates(config, path, data)
        if not path:
            return fail("缺少媒体路径")
        job = self.services.scheduler.enqueue(
            title=str(data.get("title") or Path(path).stem),
            path=path,
            identity={
                "media_source": str(data.get("media_source") or ""),
                "media_id": str(data.get("media_id") or ""),
                "tmdbid": str(data.get("tmdbid") or ""),
                "doubanid": str(data.get("doubanid") or ""),
            },
            trigger="manual",
            priority=str(data.get("priority") or "P0"),
            strategy=str(data.get("strategy") or config.get("transfer_strategy") or "search_then_translate"),
            config=config,
            payload=data,
            force=bool(data.get("force") or not ok_gate and data.get("force_skip_gate")),
        )
        if job.status == "skipped":
            return fail(job.error or reason, job.to_dict())
        return ok(job.to_dict(), "已入队")

    def api_get_job(self, job_id: str) -> Dict[str, Any]:
        job = self.services.store.get_job(job_id)
        if not job:
            return fail("任务不存在")
        return ok(job.to_dict())

    def api_cut_in(self, job_id: str) -> Dict[str, Any]:
        job = self.services.scheduler.cut_in(job_id)
        if not job:
            return fail("任务不存在")
        if job.status != "pending" and job.priority != "P0":
            return fail("运行中的任务不能插队", job.to_dict())
        return ok(job.to_dict(), "已插到队首")

    def api_set_priority(self, job_id: str, body: Dict[str, Any] = None) -> Dict[str, Any]:
        job = self.services.scheduler.set_priority(job_id, str((body or {}).get("priority") or ""))
        if not job:
            return fail("任务不存在")
        return ok(job.to_dict(), "已改优先级")

    def api_cancel_job(self, job_id: str) -> Dict[str, Any]:
        job = self.services.scheduler.cancel(job_id)
        if not job:
            return fail("任务不存在")
        return ok(job.to_dict(), "已取消")

    def api_retry_job(self, job_id: str) -> Dict[str, Any]:
        job = self.services.scheduler.retry(job_id)
        if not job:
            return fail("任务不存在")
        return ok(job.to_dict(), "已重新入队")

    def api_get_cues(self, job_id: str) -> Dict[str, Any]:
        graph = self.services.store.get_graph(job_id)
        if not graph:
            return ok({"job_id": job_id, "cues": [], "notes": [], "briefing": []})
        return ok(graph.to_dict())

    def api_save_cue(self, job_id: str, cue_id: str, body: Dict[str, Any] = None) -> Dict[str, Any]:
        graph = self.services.store.get_graph(job_id)
        if not graph:
            return fail("没有可编辑的字幕图")
        cue = graph.replace_cue(cue_id, **(body or {}))
        if not cue:
            return fail("句子不存在")
        self.services.store.save_graph(graph)
        return ok(cue.to_dict(), "已改句")

    def api_export_job(self, job_id: str) -> Dict[str, Any]:
        job = self.services.store.get_job(job_id)
        graph = self.services.store.get_graph(job_id)
        if not job or not graph:
            return fail("没有可导出的任务")
        from ..packager.export_pack import write_export_pack
        written = write_export_pack(normalize_plugin_config(self.get_config() or {}), job.path, graph)
        return ok({"items": written}, "已按当前 CueGraph 写盘")

    def api_search_job(self, job_id: str) -> Dict[str, Any]:
        job = self.services.store.get_job(job_id)
        if not job:
            return fail("任务不存在")
        results = self.services.searcher.search(job, normalize_plugin_config(self.get_config() or {}))
        return ok({"items": results})

    def api_test_endpoint(self, body: Dict[str, Any] = None) -> Dict[str, Any]:
        data = body or {}
        url = str(data.get("api_url") or "").rstrip("/")
        if not url:
            return fail("缺少 API URL")
        compatible = bool(data.get("compatible"))
        probe = url if compatible else f"{url}/v1/models"
        try:
            payload = self.services.http("GET", probe, headers=_auth_header(data.get("api_key")), use_proxy=bool(data.get("use_proxy")))
        except Exception as exc:  # noqa: BLE001
            return fail(str(exc))
        return ok({"ok": True, "payload": _safe_payload(payload)}, "连通")

    def api_list_models(self, body: Dict[str, Any] = None) -> Dict[str, Any]:
        data = body or {}
        url = str(data.get("api_url") or "").rstrip("/")
        if not url:
            return fail("缺少 API URL")
        compatible = bool(data.get("compatible"))
        probe = url if compatible else f"{url}/v1/models"
        try:
            payload = self.services.http("GET", probe, headers=_auth_header(data.get("api_key")), use_proxy=bool(data.get("use_proxy")))
        except Exception as exc:  # noqa: BLE001
            return fail(str(exc))
        models = []
        rows = payload.get("data") if isinstance(payload, dict) else []
        for item in rows or []:
            if isinstance(item, dict) and item.get("id"):
                models.append(str(item["id"]))
        return ok({"items": models})

    def api_preview_ass(self, job_id: str):
        graph = self.services.store.get_graph(job_id)
        config = normalize_plugin_config(self.get_config() or {})
        langs = list(config.get("target_languages") or ["zh-Hans"])
        text = render_ass(graph, langs, stack=config.get("lang_stack") or "main_bottom") if graph else ""
        return _plain_response(text, "text/plain; charset=utf-8")

    def api_preview_video(self, job_id: str, request=None):
        job = self.services.store.get_job(job_id)
        if not job or not job.path or not Path(job.path).is_file() or job.is_strm:
            return _empty_video()
        return _range_file(Path(job.path), request)


def _auth_header(key: Any) -> Dict[str, str]:
    token = str(key or "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def _safe_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {key: value for key, value in payload.items() if key.lower() not in {"key", "token", "authorization"}}
    return True


def _plain_response(text: str, content_type: str):
    try:
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(text or "", media_type=content_type)
    except Exception:
        return text


def _empty_video():
    try:
        from fastapi.responses import Response
        return Response(status_code=204)
    except Exception:
        return None


def _range_file(path: Path, request):
    try:
        from fastapi.responses import FileResponse
        return FileResponse(path, filename=path.name, media_type="video/mp4")
    except Exception:
        return str(path)
