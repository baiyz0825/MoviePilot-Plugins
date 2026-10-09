"""队列调度。

优先级 P0 > P1 > P2。插队 = 升 P0 + 排到 pending 队首，不打断 running。
错峰只约束 P2；人工 P0 和入库 P1 随时可跑。
"""

from __future__ import annotations

import threading
import traceback
import uuid
from datetime import datetime, time as dt_time
from typing import Any, Callable, Dict, Optional

from ..core.logging import studio_done, studio_log
from ..core.models import Job
from ..core.notify import should_notify
from ..ingest.debounce import IngestDebouncer
from ..ingest.gates import evaluate_gates, is_strm_path
from ..storage.job_store import JobStore, utc_now


def parse_offpeak_window(text: str) -> Optional[tuple]:
    raw = str(text or "").replace("–", "-").replace("—", "-").strip()
    if not raw or "-" not in raw:
        return None
    start_text, end_text = [part.strip() for part in raw.split("-", 1)]
    try:
        start = datetime.strptime(start_text, "%H:%M").time()
        end = datetime.strptime(end_text, "%H:%M").time()
    except ValueError:
        return None
    return start, end


def in_offpeak_window(window: Optional[tuple], now: Optional[dt_time] = None) -> bool:
    if not window:
        return False
    start, end = window
    current = now or datetime.now().time()
    if start <= end:
        return start <= current <= end
    return current >= start or current <= end


class JobScheduler:
    def __init__(
        self,
        store: JobStore,
        runner: Callable[[Job], None],
        *,
        logger: Any = None,
        notify: Optional[Callable[[Job], None]] = None,
    ):
        self.store = store
        self.runner = runner
        self.logger = logger
        self.notify = notify
        self.debouncer = IngestDebouncer()
        self._stop = threading.Event()
        self._worker: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    def start(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._stop.clear()
        self._worker = threading.Thread(target=self._loop, name="subtitlestudio-queue", daemon=True)
        self._worker.start()

    def stop(self) -> None:
        self._stop.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=2)
        self._worker = None

    def enqueue(
        self,
        *,
        title: str,
        path: str,
        identity: Optional[Dict[str, str]] = None,
        trigger: str = "manual",
        priority: str = "P1",
        strategy: str = "search_then_translate",
        config: Optional[Dict[str, Any]] = None,
        payload: Optional[Dict[str, Any]] = None,
        force: bool = False,
    ) -> Job:
        identity = identity or {}
        if config is None:
            getter = getattr(self, "config_getter", None)
            config = getter() if callable(getter) else {}
        ok, reason = evaluate_gates(config or {}, path, payload)
        if not ok and not force:
            studio_log(self.logger, "info", "入队跳过 title=%s trigger=%s reason=%s path=%s", title or path, trigger, reason, path)
            job = Job(
                job_id=uuid.uuid4().hex,
                title=title or path,
                path=path,
                priority=priority,
                status="skipped",
                trigger=trigger,
                strategy=strategy,
                is_strm=is_strm_path(path),
                error=reason,
                queue_rank=self.store.tail_queue_rank(),
                **{key: identity.get(key, "") for key in ("media_source", "media_id", "tmdbid", "doubanid")},
                payload=payload or {},
            )
            saved = self.store.save_job(job)
            self._maybe_notify(saved)
            return saved
        if trigger != "manual" and not force and not self.debouncer.allow(identity, path):
            studio_log(self.logger, "info", "5 分钟防抖，跳过重复入队 trigger=%s path=%s", trigger, path)
            existing = next((item for item in self.store.list_jobs(q=path, limit=20) if item.path == path), None)
            if existing:
                return existing
        job = Job(
            job_id=uuid.uuid4().hex,
            title=title or path,
            path=path,
            priority=priority if priority in {"P0", "P1", "P2"} else "P1",
            status="pending",
            trigger=trigger,
            strategy=strategy,
            is_strm=is_strm_path(path),
            queue_rank=self.store.tail_queue_rank(),
            media_source=identity.get("media_source", ""),
            media_id=identity.get("media_id", ""),
            tmdbid=identity.get("tmdbid", ""),
            doubanid=identity.get("doubanid", ""),
            payload=payload or {},
        )
        saved = self.store.save_job(job)
        studio_log(
            self.logger,
            "info",
            "已入队 job=%s title=%s trigger=%s priority=%s strategy=%s strm=%s path=%s",
            saved.job_id,
            saved.title,
            saved.trigger,
            saved.priority,
            saved.strategy,
            saved.is_strm,
            saved.path,
        )
        self.kick()
        return saved

    def cut_in(self, job_id: str) -> Optional[Job]:
        job = self.store.get_job(job_id)
        if not job or job.status != "pending":
            return job
        job.priority = "P0"
        job.queue_rank = self.store.next_queue_rank()
        studio_log(self.logger, "info", "插队 job=%s title=%s 升为 P0", job.job_id, job.title)
        return self.store.save_job(job)

    def set_priority(self, job_id: str, priority: str) -> Optional[Job]:
        job = self.store.get_job(job_id)
        if not job or job.status not in {"pending"} or priority not in {"P0", "P1", "P2"}:
            return job
        job.priority = priority
        studio_log(self.logger, "info", "改优先级 job=%s title=%s -> %s", job.job_id, job.title, priority)
        return self.store.save_job(job)

    def cancel(self, job_id: str) -> Optional[Job]:
        job = self.store.get_job(job_id)
        if not job or job.status not in {"pending", "running"}:
            return job
        if job.status == "running":
            job.payload = {**job.payload, "cancel": True}
            studio_log(self.logger, "info", "标记取消运行中任务 job=%s title=%s", job.job_id, job.title)
            return self.store.save_job(job)
        job.status = "cancelled"
        job.finished_at = utc_now()
        studio_log(self.logger, "info", "取消排队任务 job=%s title=%s", job.job_id, job.title)
        saved = self.store.save_job(job)
        self._maybe_notify(saved)
        return saved

    def retry(self, job_id: str) -> Optional[Job]:
        job = self.store.get_job(job_id)
        if not job:
            return None
        job.status = "pending"
        job.error = ""
        job.finished_at = ""
        job.started_at = ""
        job.payload = {k: v for k, v in job.payload.items() if k != "cancel"}
        job.queue_rank = self.store.tail_queue_rank()
        saved = self.store.save_job(job)
        studio_log(self.logger, "info", "重试入队 job=%s title=%s", saved.job_id, saved.title)
        self.kick()
        return saved

    def kick(self) -> None:
        if not self._worker or not self._worker.is_alive():
            self.start()

    def _loop(self) -> None:
        while not self._stop.wait(0.4):
            try:
                self._step()
            except Exception as exc:  # noqa: BLE001 — 工作线程不能被单次失败打死
                studio_log(self.logger, "error", "队列循环失败：%s", exc)
                if self.logger:
                    self.logger.error(traceback.format_exc())

    def _step(self) -> None:
        if self.store.running_job():
            return
        config = {}
        allow_p2 = True
        getter = getattr(self, "config_getter", None)
        if callable(getter):
            config = getter() or {}
            window = parse_offpeak_window(config.get("queue_offpeak_window") or "")
            if config.get("queue_offpeak_enabled"):
                allow_p2 = in_offpeak_window(window)
        pending = self.store.pending_jobs(allow_p2=allow_p2)
        if not pending:
            return
        job = pending[0]
        job.status = "running"
        job.started_at = utc_now()
        self.store.save_job(job)
        studio_log(self.logger, "info", "开始执行 job=%s title=%s priority=%s", job.job_id, job.title, job.priority)
        try:
            self.runner(job)
            latest = self.store.get_job(job.job_id) or job
            if latest.payload.get("cancel"):
                latest.status = "cancelled"
            elif latest.status == "running":
                latest.status = "success"
            latest.finished_at = utc_now()
            self.store.save_job(latest)
            studio_log(self.logger, "info", "队列回收 job=%s status=%s", latest.job_id, latest.status)
        except Exception as exc:  # noqa: BLE001
            job.status = "failed"
            job.error = str(exc)
            job.finished_at = utc_now()
            self.store.save_job(job)
            studio_log(self.logger, "error", "任务失败 job=%s title=%s error=%s", job.job_id, job.title, exc)
            if self.logger:
                self.logger.error(traceback.format_exc())
            self._maybe_notify(job)
            studio_done(self.logger)

    def _maybe_notify(self, job: Job) -> None:
        getter = getattr(self, "config_getter", None)
        config = getter() if callable(getter) else {}
        if not should_notify(config or {}, job) or not self.notify:
            return
        try:
            self.notify(job)
        except Exception as exc:  # noqa: BLE001
            studio_log(self.logger, "warning", "通知推送失败 job=%s：%s", job.job_id, exc)
