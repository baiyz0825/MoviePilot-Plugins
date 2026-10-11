"""生成流水线：搜 →（可选）ASR → 译 → 特效 → 导出。

STRM 永远不做 ASR 和调轴。能搜到外挂就不再 ASR。
不委托 AutoSub / 海拉鲁。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..core.cuegraph import parse_subtitle
from ..core.embedded import extract_best_embedded
from ..core.web_research import web_research
from ..core.format_repair import repair_asr_graph
from ..core.logging import studio_done, studio_log, studio_step
from ..core.models import CueGraph, Job
from ..core.notify import should_notify
from ..ingest.gates import is_strm_path
from ..packager.export_pack import write_export_pack
from ..storage.job_store import JobStore
from .effects import EffectsBuilder
from .translation import TranslationService


class GenerationPipeline:
    def __init__(
        self,
        store: JobStore,
        config_getter: Callable[[], Dict[str, Any]],
        *,
        searcher: Optional[Callable[..., Optional[Dict[str, Any]]]] = None,
        asr: Optional[Callable[..., Optional[CueGraph]]] = None,
        http: Callable[..., Any] = lambda *args, **kwargs: None,
        llm: Optional[Callable[..., str]] = None,
        logger: Any = None,
        notify: Optional[Callable[[Job], None]] = None,
    ):
        self.store = store
        self.config_getter = config_getter
        self.searcher = searcher
        self.asr = asr
        self.http = http
        self.llm = llm
        self.logger = logger
        self.notify = notify
        self._current_job: Optional[Job] = None

    STAGE_MAP = {
        0: ("init", 5, "准备中"),
        1: ("local_sidecar", 12, "查找本地外挂"),
        2: ("search", 25, "在线搜索"),
        3: ("embedded", 40, "提取内嵌字幕"),
        4: ("asr", 55, "语音识别"),
        5: ("translate", 72, "翻译对白"),
        6: ("effects", 86, "生成特效"),
        7: ("export", 95, "写出导出包"),
    }

    def _progress(self, step: int, message: str, *args) -> None:
        studio_log(self.logger, "info", f"[Step {step}] {message}", *args)
        stage, percent, detail = self.STAGE_MAP.get(step, ("", 0, ""))
        if stage and self._current_job:
            self._current_job.progress = percent
            self._current_job.stage = stage
            self._current_job.stage_detail = detail
            try:
                self.store.save_job(self._current_job)
            except Exception:
                pass

    def run(self, job: Job) -> Job:
        started = time.monotonic()
        config = self.config_getter() or {}
        latest = self.store.get_job(job.job_id) or job
        self._current_job = latest
        self._progress(
            0,
            "开始处理 title=%s job=%s trigger=%s priority=%s strategy=%s strm=%s path=%s",
            latest.title,
            latest.job_id,
            latest.trigger,
            latest.priority,
            latest.strategy or config.get("transfer_strategy") or "search_then_translate",
            latest.is_strm,
            latest.path,
        )
        if latest.payload.get("cancel"):
            latest.status = "cancelled"
            studio_log(self.logger, "info", "用户取消，停止处理 job=%s", latest.job_id)
            return self._finish(latest, config, started)
        strategy = latest.strategy or config.get("transfer_strategy") or "search_then_translate"
        graph, asr_ran, asr_graph = self._resolve_source(latest, config, strategy)
        if graph is None:
            latest.status = "failed"
            latest.error = latest.error or "没有可用字幕源"
            studio_log(self.logger, "error", "没有可用字幕源 job=%s path=%s", latest.job_id, latest.path)
            return self._finish(latest, config, started)
        studio_log(
            self.logger,
            "info",
            "字幕源就绪 cues=%s notes=%s lang=%s asr=%s",
            len(graph.cues),
            len(graph.notes),
            graph.source_lang,
            asr_ran,
        )
        translator = TranslationService(config, http=self.http, llm=self.llm, logger=self.logger)
        if strategy != "search_only" and config.get("translate_enabled", True):
            langs = config.get("target_languages") or ["zh-Hans"]
            self._progress(5, "翻译开始 langs=%s backend=%s cues=%s", langs, config.get("translate_backend") or "free_first", len(graph.cues))
            graph = translator.translate_graph(graph, langs)
            rate = translator.failure_rate(graph, langs[0])
            studio_log(self.logger, "info", "翻译结束 失败率=%.1f%%", rate * 100)
            if config.get("abort_on_high_failure") and rate > 0.30:
                latest.status = "failed"
                latest.error = "失败率过高，整片不写"
                self.store.save_graph(graph)
                studio_log(self.logger, "error", "失败率过高，整片不写 job=%s rate=%.1f%%", latest.job_id, rate * 100)
                return self._finish(latest, config, started)
        else:
            self._progress(5, "跳过翻译 strategy=%s enabled=%s", strategy, config.get("translate_enabled", True))
        if config.get("effects_enabled"):
            self._progress(6, "生成特效顶注 title=%s", latest.title)
            research = (lambda prompt, system="", role="research": self.llm(prompt, system=system, role=role)) if self.llm else None
            web_search = (lambda term, title: web_research(term, title, self.http, logger=self.logger)) if self.http else None
            graph = EffectsBuilder(config, research=research, web_search=web_search, logger=self.logger).apply(graph, title=latest.title)
            studio_log(self.logger, "info", "特效完成 notes=%s briefing=%s", len(graph.notes), len(graph.briefing))
        else:
            self._progress(6, "特效关闭，跳过顶注")
        self.store.save_graph(graph)
        if latest.path:
            self._progress(7, "写出导出包 path=%s", latest.path)
            written = write_export_pack(
                config,
                latest.path,
                graph,
                asr_graph=asr_graph,
                asr_ran=asr_ran,
            )
            latest.payload = {**latest.payload, "export": written, "asr_ran": asr_ran}
            for item in written:
                if item.get("written"):
                    studio_log(self.logger, "info", "写出 %s", item.get("filename"))
                else:
                    studio_log(self.logger, "info", "跳过已存在 %s", item.get("filename"))
        refreshed = self.store.get_job(latest.job_id) or latest
        if refreshed.payload.get("cancel"):
            latest.payload = refreshed.payload
            latest.status = "cancelled"
        else:
            latest.status = "success"
        return self._finish(latest, config, started)

    def _finish(self, latest: Job, config: Dict[str, Any], started: float) -> Job:
        if latest.status == "success":
            latest.progress = 100
            latest.stage_detail = "已完成"
        self._current_job = None
        saved = self.store.save_job(latest)
        self._emit_notify(saved, config)
        studio_log(
            self.logger,
            "info",
            "处理完成 title=%s job=%s status=%s 耗时=%.2f秒",
            saved.title,
            saved.job_id,
            saved.status,
            time.monotonic() - started,
        )
        studio_done(self.logger)
        return saved

    def _emit_notify(self, job: Job, config: Dict[str, Any]) -> None:
        if not should_notify(config, job) or not self.notify:
            return
        try:
            self.notify(job)
        except Exception as exc:  # noqa: BLE001 — 通知失败不能把任务改成失败
            studio_log(self.logger, "warning", "通知推送失败 job=%s：%s", job.job_id, exc)

    def _resolve_source(self, job: Job, config: Dict[str, Any], strategy: str):
        asr_ran = False
        asr_graph = None
        self._progress(1, "查找本地外挂 path=%s", job.path)
        graph = self._load_local_sidecar(job)
        if graph:
            studio_log(self.logger, "info", "使用本地外挂 cues=%s", len(graph.cues))
        elif strategy != "translate_only" and self.searcher:
            self._progress(2, "在线搜索 title=%s", job.title)
            found = self.searcher(job, config)
            if found and found.get("content"):
                graph = parse_subtitle(found["content"], found.get("filename") or "", lang=found.get("lang") or "source", job_id=job.job_id)
                studio_log(self.logger, "info", "采用在线字幕 file=%s lang=%s cues=%s", found.get("filename") or found.get("title"), found.get("lang"), len(graph.cues))
            else:
                studio_log(self.logger, "info", "在线搜索没有可用正文")
        else:
            self._progress(2, "跳过在线搜索 strategy=%s", strategy)
        if graph is None and strategy != "search_only" and not (job.is_strm or is_strm_path(job.path)):
            self._progress(3, "提取内嵌字幕 path=%s", job.path)
            graph = extract_best_embedded(job, config, self.logger)
            if graph:
                studio_log(self.logger, "info", "使用内嵌字幕 cues=%s", len(graph.cues))
        if graph is None and strategy != "search_only" and config.get("enable_asr") and not (job.is_strm or is_strm_path(job.path)):
            self._progress(4, "开始 ASR model=%s", config.get("whisper_model") or "base")
            if self.asr:
                graph = self.asr(job, config)
                if graph:
                    asr_ran = True
                    graph = repair_asr_graph(graph, enabled=bool(config.get("asr_repair_enabled", True)))
                    asr_graph = CueGraph.from_dict(graph.to_dict())
                    studio_log(self.logger, "info", "ASR 完成 cues=%s lang=%s", len(graph.cues), graph.source_lang)
                else:
                    studio_log(self.logger, "warning", "ASR 没有产出字幕")
        elif graph is None:
            reason = "STRM 不做识别" if (job.is_strm or is_strm_path(job.path)) else ("策略只搜索" if strategy == "search_only" else "ASR 未开启")
            self._progress(4, "跳过 ASR：%s", reason)
        if graph:
            graph.job_id = job.job_id
        return graph, asr_ran, asr_graph

    def _load_local_sidecar(self, job: Job) -> Optional[CueGraph]:
        path = Path(job.path)
        if not path.parent.is_dir():
            studio_log(self.logger, "info", "视频目录不存在，跳过本地外挂 %s", path.parent)
            return None
        for item in sorted(path.parent.iterdir()):
            if item.suffix.lower() not in {".srt", ".ass", ".ssa", ".vtt"}:
                continue
            if path.stem.lower() not in item.name.lower():
                continue
            try:
                studio_log(self.logger, "info", "发现本地外挂 %s", item.name)
                return parse_subtitle(item.read_text(encoding="utf-8", errors="ignore"), item.name, job_id=job.job_id)
            except OSError as exc:
                studio_log(self.logger, "warning", "读取本地外挂失败 %s：%s", item.name, exc)
                continue
        return None
