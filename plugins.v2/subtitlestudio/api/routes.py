"""四个一级页共用的 API。

写操作和列表走同一 envelope。预览视频 / ASS 是原生流，不要包信封。
不要设匿名。页面调用 auth=bear。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from ..core.config_schema import FIELDS, PANES, apply_export_preset, default_config, normalize_plugin_config
from ..core.cuegraph import render_ass
from ..core.embedded import probe_video_resolution
from ..core.errors import user_message
from ..core.style import style_from_config
from ..core.history import group_media_items
from ..core.naming import extra_tracks, export_plan, sanitize_stem
from ..core.response import fail, ok
from ..ingest.gates import evaluate_gates, is_strm_path
from .schemas import make_envelope, payload_model_for

_SECRET_KEYS = {"assrt_api_key", "opensubtitles_api_key", "api_key"}
_MEDIA_SUFFIXES = {".mkv", ".mp4", ".avi", ".mov", ".m4v", ".ts", ".m2ts", ".wmv", ".flv", ".webm", ".strm"}


def build_api_routes(plugin) -> List[Dict[str, Any]]:
    return [
        _route("/status", plugin.api_status, ["GET"], "插件状态"),
        _route("/config", plugin.api_get_config, ["GET"], "读取配置"),
        _route("/config", plugin.api_save_config, ["POST"], "保存配置"),
        _route("/fields", plugin.api_fields, ["GET"], "设置页字段合同"),
        _route("/media", plugin.api_list_media, ["GET"], "媒体目录"),
        _route("/media/refresh", plugin.api_refresh_media, ["POST"], "拉取整理记录"),
        _route("/media/scan", plugin.api_start_stock_scan, ["POST"], "扫描存量缺失字幕"),
        _route("/media/scan", plugin.api_stock_scan_status, ["GET"], "读取存量扫描进度"),
        _route("/media/hide", plugin.api_hide_media, ["POST"], "从列表移除媒体"),
        _route("/jobs", plugin.api_list_jobs, ["GET"], "任务队列"),
        _route("/jobs", plugin.api_create_job, ["POST"], "入队"),
        _route("/jobs/batch", plugin.api_create_jobs, ["POST"], "批量入队"),
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
        _preview_route("/jobs/{job_id}/preview/video", plugin.api_preview_video, "预览原片（Range）", "video"),
        _preview_route("/jobs/{job_id}/preview/ass", plugin.api_preview_ass, "当前 CueGraph 打成预览 ASS", "ass"),
    ]


def finalize_api_routes(routes: List[Dict[str, Any]], *, generation: str) -> List[Dict[str, Any]]:
    """V3 用宿主 Response[T]；V2 用本地信封模型。原生预览保持 response_model=None。"""
    host_response = None
    try:
        from app.schemas import Response
        host_response = Response
    except Exception:
        host_response = None
    finalized = []
    for item in routes:
        route = dict(item)
        if route.get("path", "").endswith(("/preview/video", "/preview/ass")):
            finalized.append(route)
            continue
        payload = payload_model_for(route.get("path") or "", (route.get("methods") or ["GET"])[0])
        if payload is None:
            finalized.append(route)
            continue
        if generation == "v3" and host_response is not None:
            try:
                route["response_model"] = host_response[payload]
            except Exception:
                route["response_model"] = make_envelope(payload) or payload
        else:
            route["response_model"] = make_envelope(payload) or payload
        finalized.append(route)
    return finalized


def _route(path: str, endpoint, methods: List[str], summary: str) -> Dict[str, Any]:
    return {
        "path": path,
        "endpoint": endpoint,
        "methods": methods,
        "auth": "bear",
        "summary": summary,
    }


def _preview_route(path: str, endpoint, summary: str, kind: str) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "path": path,
        "endpoint": endpoint,
        "methods": ["GET"],
        "auth": "bear",
        "summary": summary,
        "response_model": None,
    }
    if kind == "ass":
        try:
            from fastapi.responses import PlainTextResponse
            item["response_class"] = PlainTextResponse
        except Exception:
            pass
        item["responses"] = {200: {"content": {"text/plain": {"schema": {"type": "string"}}}}}
        return item
    try:
        from fastapi.responses import FileResponse
        item["response_class"] = FileResponse
    except Exception:
        pass
    item["responses"] = {
        200: {"content": {"video/mp4": {"schema": {"type": "string", "format": "binary"}}}},
        204: {"description": "STRM 或无法提供原片"},
    }
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
        return ok(_mask_config(normalize_plugin_config(self.get_config() or {})))

    def api_save_config(self, body: Dict[str, Any] = None) -> Dict[str, Any]:
        current = normalize_plugin_config(self.get_config() or {})
        payload = normalize_plugin_config(body or {})
        payload = _restore_masked_secrets(payload, current)
        if payload.get("export_preset"):
            payload = apply_export_preset(payload, payload["export_preset"])
        self.update_config(payload)
        self.init_plugin(payload)
        return ok(_mask_config(payload), "已保存")

    def api_fields(self) -> Dict[str, Any]:
        return ok({"fields": FIELDS, "panes": PANES, "defaults": default_config()})

    def api_list_media(self, q: str = "", media_type: str = "", force: bool = False) -> Dict[str, Any]:
        items = self.services.catalog.list_media(q=q, media_type=media_type, force=force)
        hidden = self.services.store.hidden_paths()
        if hidden:
            items = [item for item in items if item.get("path") not in hidden]
        groups = group_media_items(items)
        successful_paths = self.services.store.successful_paths(
            file.get("path") for group in groups for file in group.get("files") or []
        )
        for group in groups:
            for file in group.get("files") or []:
                file["already_processed"] = file.get("path") in successful_paths
            group["processed_count"] = sum(
                bool(file.get("already_processed")) for file in group.get("files") or []
            )
        return ok({
            "groups": groups,
            "counts": {"files": len(items), "groups": len(groups)},
            "source": "transfer_history",
        })

    def api_refresh_media(self) -> Dict[str, Any]:
        return self.api_list_media(force=True)

    def api_start_stock_scan(self) -> Dict[str, Any]:
        state = self.services.stock_scan.start()
        message = "存量字幕扫描已启动" if state.get("started") else "存量字幕扫描正在运行"
        return ok(state, message)

    def api_stock_scan_status(self) -> Dict[str, Any]:
        return ok(self.services.stock_scan.status())

    def api_hide_media(self, body: Dict[str, Any] = None) -> Dict[str, Any]:
        data = body or {}
        paths = []
        seen = set()
        for raw in data.get("paths") or []:
            path = str(raw or "").strip()
            if not path or path in seen or not _hideable_media_path(path):
                continue
            seen.add(path)
            paths.append(path)
        if not paths:
            return fail("没有可移除的媒体")
        deleted: List[str] = []
        if data.get("delete_subtitles"):
            config = normalize_plugin_config(self.get_config() or {})
            from ..packager.export_pack import delete_generated_subtitles
            for path in paths:
                deleted.extend(delete_generated_subtitles(config, path))
        hidden = self.services.store.hide_paths(paths)
        self.services.catalog.invalidate()
        return ok({"hidden": hidden, "deleted_files": deleted}, "已从列表移除")

    def api_list_jobs(self, q: str = "", status: str = "") -> Dict[str, Any]:
        jobs = [item.to_dict() for item in self.services.store.list_jobs(q=q, status=status)]
        return ok({"items": jobs, "counts": self.services.store.counts()})

    def api_create_job(self, body: Dict[str, Any] = None) -> Dict[str, Any]:
        data = body or {}
        path = str(data.get("path") or "")
        config = normalize_plugin_config(self.get_config() or {})
        if not path:
            return fail("缺少媒体路径")
        if not _path_allowed(config, path):
            return fail("媒体路径不在允许范围内")
        force = bool(data.get("force", False))
        overwrite = bool(data.get("overwrite"))
        if overwrite:
            config = {**config, "skip_existing_chinese": False}
        _ok_gate, reason = evaluate_gates(config, path, data)
        strategy = str(data.get("strategy") or "standard")
        if strategy == "search_then_translate":
            strategy = "standard"
        if is_strm_path(path) and strategy == "translate_only":
            strategy = "standard"
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
            strategy=strategy,
            config=config,
            payload={**data, "overwrite": overwrite},
            force=force,
        )
        if job.status == "skipped":
            return fail(job.error or reason, job.to_dict())
        return ok(job.to_dict(), "已入队")

    def api_create_jobs(self, body: Dict[str, Any] = None) -> Dict[str, Any]:
        data = body or {}
        rows = data.get("items") or []
        if not rows:
            return fail("没有勾选媒体文件")
        created = []
        skipped = []
        for item in rows:
            if isinstance(item, str):
                item = {"path": item}
            payload = {
                **item,
                "strategy": data.get("strategy") or item.get("strategy"),
                "priority": data.get("priority") or item.get("priority") or "P0",
                "force": bool(data.get("force", False)),
                "overwrite": bool(data.get("overwrite", False)),
            }
            result = self.api_create_job(payload)
            envelope = result if isinstance(result, dict) else {}
            job = envelope.get("data")
            if envelope.get("success") and job:
                created.append(job)
            else:
                skipped.append({"path": item.get("path"), "reason": envelope.get("message") or "跳过"})
        message = f"已入队 {len(created)} 条"
        if skipped:
            message += f"，跳过 {len(skipped)} 条"
        return ok({"items": created, "skipped": skipped, "count": len(created)}, message)

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

    def api_retry_job(self, job_id: str, body: Dict[str, Any] = None) -> Dict[str, Any]:
        overwrite = bool((body or {}).get("overwrite"))
        job = self.services.scheduler.retry(job_id, overwrite=overwrite)
        if not job:
            return fail("任务不存在")
        return ok(job.to_dict(), "已重新入队，将覆盖已有字幕" if overwrite else "已重新入队")

    def api_get_cues(self, job_id: str) -> Dict[str, Any]:
        graph = self.services.store.get_graph(job_id)
        if not graph:
            return ok({"job_id": job_id, "cues": [], "notes": [], "briefing": []})
        return ok(graph.to_dict())

    def api_save_cue(self, job_id: str, cue_id: str, body: Dict[str, Any] = None) -> Dict[str, Any]:
        graph = self.services.store.get_graph(job_id)
        if not graph:
            return fail("没有可编辑的字幕图")
        data = body or {}
        allowed = {key: data[key] for key in ("texts", "start_ms", "end_ms", "kind") if key in data}
        cue = graph.replace_cue(cue_id, **allowed)
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
            payload = self.services.http(
                "GET", probe,
                headers=_auth_header(data.get("api_key")),
                use_proxy=bool(data.get("use_proxy")),
                timeout=600 if data.get("local_model") else 20,
            )
        except Exception as exc:  # noqa: BLE001
            return fail(user_message(exc, fallback="连通失败"))
        return ok({"ok": True, "payload": _safe_payload(payload)}, "连通")

    def api_list_models(self, body: Dict[str, Any] = None) -> Dict[str, Any]:
        data = body or {}
        url = str(data.get("api_url") or "").rstrip("/")
        if not url:
            return fail("缺少 API URL")
        compatible = bool(data.get("compatible"))
        probe = url if compatible else f"{url}/v1/models"
        try:
            payload = self.services.http(
                "GET", probe,
                headers=_auth_header(data.get("api_key")),
                use_proxy=bool(data.get("use_proxy")),
                timeout=600 if data.get("local_model") else 20,
            )
        except Exception as exc:  # noqa: BLE001
            return fail(user_message(exc, fallback="拉取模型失败"))
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
        job = self.services.store.get_job(job_id)
        play_width, play_height = probe_video_resolution(job.path, logger=None) if job and job.path else (0, 0)
        text = render_ass(
            graph,
            langs,
            stack=config.get("lang_stack") or "main_bottom",
            style=style_from_config(config),
            play_width=play_width,
            play_height=play_height,
        ) if graph else ""
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


def _mask_config(config: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(config or {})
    for key in _SECRET_KEYS:
        if out.get(key):
            out[key] = "********"
    rows = []
    for item in out.get("openai_endpoints") or []:
        if not isinstance(item, dict):
            continue
        row = dict(item)
        if row.get("api_key"):
            row["api_key"] = "********"
        rows.append(row)
    if "openai_endpoints" in out:
        out["openai_endpoints"] = rows
    return out


def _restore_masked_secrets(payload: Dict[str, Any], current: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload or {})
    for key in _SECRET_KEYS:
        if out.get(key) in {"", "********"} and current.get(key):
            if out.get(key) == "********":
                out[key] = current.get(key)
    current_eps = {str(item.get("endpoint_id")): item for item in (current.get("openai_endpoints") or []) if isinstance(item, dict)}
    rows = []
    for item in out.get("openai_endpoints") or []:
        if not isinstance(item, dict):
            continue
        row = dict(item)
        if row.get("api_key") == "********":
            old = current_eps.get(str(row.get("endpoint_id"))) or {}
            row["api_key"] = old.get("api_key") or ""
        rows.append(row)
    if "openai_endpoints" in out:
        out["openai_endpoints"] = rows
    return out


def _hideable_media_path(path: str) -> bool:
    raw = str(path or "").strip()
    if not raw or ".." in Path(raw).parts:
        return False
    return Path(raw).suffix.lower() in _MEDIA_SUFFIXES


def _path_allowed(config: Dict[str, Any], path: str) -> bool:
    raw = str(path or "").strip()
    if not raw or ".." in Path(raw).parts:
        return False
    try:
        resolved = Path(raw).resolve()
    except OSError:
        return False
    suffix = resolved.suffix.lower()
    if suffix in {".txt", ".json", ".py", ".sh", ".exe", ".dll", ".so", ".conf", ".env"}:
        return False
    # 整理记录里的真实媒体文件直接放行；否则要求落在监控根下
    if resolved.is_file() and (suffix in _MEDIA_SUFFIXES or not suffix):
        return True
    roots: List[Path] = []
    for blob in (config.get("watch_paths"), config.get("strm_paths")):
        for line in str(blob or "").splitlines():
            line = line.strip()
            if line:
                try:
                    roots.append(Path(line).resolve())
                except OSError:
                    continue
    if not roots:
        return suffix in _MEDIA_SUFFIXES or not suffix
    return any(resolved == root or root in resolved.parents for root in roots)


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
