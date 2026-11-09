"""运行时装配。init_plugin 里才建，导入期不启动线程、不访问网络。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .core.logging import studio_log
from .ingest.stock_scan import StockSubtitleScan
from .ingest.watch import WatchService, strm_paths_from_config, watch_paths_from_config
from .pipeline.asr import transcribe
from .pipeline.generation import GenerationPipeline
from .pipeline.llm import LlmRouter
from .pipeline.scheduler import JobScheduler
from .providers.catalog import MediaCatalog
from .providers.online import OnlineSearchService
from .storage.job_store import JobStore


class StudioServices:
    def __init__(self, plugin):
        self.plugin = plugin
        data_path = Path(str(plugin.get_data_path()))
        self.store = JobStore(data_path, plugin_id=plugin.__class__.__name__)
        self.http = plugin.host_http
        self.catalog = MediaCatalog(
            plugin.current_config,
            history_loader=plugin.host_history,
            library_roots=plugin.host_library_roots,
            logger=plugin.host_logger,
        )
        self.searcher = OnlineSearchService(self.http, plugin.current_config(), plugin.host_logger)
        self.llm = LlmRouter(self.http, plugin.current_config(), plugin.host_logger)

        def runner(job):
            pipeline.run(job)

        self.scheduler = JobScheduler(self.store, runner, logger=plugin.host_logger, notify=plugin.notify_job)
        self.scheduler.config_getter = plugin.current_config
        pipeline = GenerationPipeline(
            self.store,
            plugin.current_config,
            searcher=self.searcher.best_download,
            asr=lambda job, config, cancel_check=None: transcribe(
                job, config, plugin.host_logger, cancel_check=cancel_check
            ),
            http=self.http,
            llm=self.llm.complete,
            history_loader=plugin.host_history,
            library_roots=plugin.host_library_roots(),
            logger=plugin.host_logger,
            notify=plugin.notify_job,
        )
        self.pipeline = pipeline
        self.stock_scan = StockSubtitleScan(
            self.catalog, self.scheduler, self.store, plugin.current_config, plugin.host_logger
        )
        self.watch = WatchService(plugin.ingest_watched_file, plugin.host_logger)
        self.strm_watch = WatchService(plugin.ingest_watched_file, plugin.host_logger)

    def start(self, config: dict) -> None:
        self.scheduler.start()
        studio_log(
            self.plugin.host_logger,
            "info",
            "队列已启动 event=%s watch=%s strm=%s queue=%s translate_workers=%s",
            config.get("ingest_on_event"),
            config.get("ingest_on_watch"),
            config.get("strm_enabled"),
            config.get("queue_workers") or 2,
            config.get("parallel_workers"),
        )
        if config.get("ingest_on_watch"):
            self.watch.start(watch_paths_from_config(config), strm=False)
        if config.get("strm_enabled"):
            self.strm_watch.start(strm_paths_from_config(config), strm=True)

    def stop(self) -> None:
        self.stock_scan.stop()
        self.scheduler.stop()
        self.watch.stop()
        self.strm_watch.stop()
