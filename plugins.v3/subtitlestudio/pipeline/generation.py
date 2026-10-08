"""生成流水线：搜 →（可选）ASR → 译 → 特效 → 导出。

STRM 永远不做 ASR 和调轴。能搜到外挂就不再 ASR。
不委托 AutoSub / 海拉鲁。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Dict, Optional

from ..core.cuegraph import parse_subtitle
from ..core.format_repair import repair_asr_graph
from ..core.models import CueGraph, Job
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

    def run(self, job: Job) -> Job:
        config = self.config_getter() or {}
        latest = self.store.get_job(job.job_id) or job
        if latest.payload.get("cancel"):
            latest.status = "cancelled"
            return self.store.save_job(latest)
        strategy = latest.strategy or config.get("transfer_strategy") or "search_then_translate"
        graph, asr_ran, asr_graph = self._resolve_source(latest, config, strategy)
        if graph is None:
            latest.status = "failed"
            latest.error = latest.error or "没有可用字幕源"
            return self.store.save_job(latest)
        translator = TranslationService(config, http=self.http, llm=self.llm, logger=self.logger)
        if strategy != "search_only" and config.get("translate_enabled", True):
            graph = translator.translate_graph(graph, config.get("target_languages") or ["zh-Hans"])
            rate = translator.failure_rate(graph, (config.get("target_languages") or ["zh-Hans"])[0])
            if config.get("abort_on_high_failure") and rate > 0.30:
                latest.status = "failed"
                latest.error = "失败率过高，整片不写"
                self.store.save_graph(graph)
                return self.store.save_job(latest)
        if config.get("effects_enabled"):
            research = (lambda prompt, system="", role="research": self.llm(prompt, system=system, role=role)) if self.llm else None
            graph = EffectsBuilder(config, research=research, logger=self.logger).apply(graph, title=latest.title)
        self.store.save_graph(graph)
        if latest.path:
            written = write_export_pack(
                config,
                latest.path,
                graph,
                asr_graph=asr_graph,
                asr_ran=asr_ran,
            )
            latest.payload = {**latest.payload, "export": written, "asr_ran": asr_ran}
        latest.status = "success"
        saved = self.store.save_job(latest)
        if config.get("send_notify") and self.notify:
            self.notify(saved)
        return saved

    def _resolve_source(self, job: Job, config: Dict[str, Any], strategy: str):
        asr_ran = False
        asr_graph = None
        graph = self._load_local_sidecar(job)
        if graph is None and strategy != "translate_only" and self.searcher:
            found = self.searcher(job, config)
            if found and found.get("content"):
                graph = parse_subtitle(found["content"], found.get("filename") or "", lang=found.get("lang") or "source", job_id=job.job_id)
        if graph is None and strategy != "search_only" and config.get("enable_asr") and not (job.is_strm or is_strm_path(job.path)):
            if self.asr:
                graph = self.asr(job, config)
                if graph:
                    asr_ran = True
                    graph = repair_asr_graph(graph, enabled=bool(config.get("asr_repair_enabled", True)))
                    asr_graph = CueGraph.from_dict(graph.to_dict())
        if graph:
            graph.job_id = job.job_id
        return graph, asr_ran, asr_graph

    def _load_local_sidecar(self, job: Job) -> Optional[CueGraph]:
        path = Path(job.path)
        if not path.parent.is_dir():
            return None
        for item in sorted(path.parent.iterdir()):
            if item.suffix.lower() not in {".srt", ".ass", ".ssa", ".vtt"}:
                continue
            if path.stem.lower() not in item.name.lower():
                continue
            try:
                return parse_subtitle(item.read_text(encoding="utf-8", errors="ignore"), item.name, job_id=job.job_id)
            except OSError:
                continue
        return None
