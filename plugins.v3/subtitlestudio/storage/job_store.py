"""Job / CueGraph 自有库。

表名带 plugin_subtitlestudio_ 前缀，行里存 plugin_id 以便分身隔离。
用标准库 sqlite3，避免领域层依赖宿主 Session。路径由宿主注入 get_data_path()。
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from ..core.models import CueGraph, Job, PRIORITY_ORDER

SCHEMA = """
CREATE TABLE IF NOT EXISTS plugin_subtitlestudio_jobs (
    job_id TEXT PRIMARY KEY,
    plugin_id TEXT NOT NULL,
    title TEXT,
    path TEXT,
    media_source TEXT,
    media_id TEXT,
    tmdbid TEXT,
    doubanid TEXT,
    priority TEXT,
    status TEXT,
    trigger TEXT,
    strategy TEXT,
    is_strm INTEGER,
    error TEXT,
    created_at TEXT,
    updated_at TEXT,
    started_at TEXT,
    finished_at TEXT,
    queue_rank INTEGER,
    payload TEXT
);
CREATE TABLE IF NOT EXISTS plugin_subtitlestudio_cues (
    job_id TEXT PRIMARY KEY,
    plugin_id TEXT NOT NULL,
    graph TEXT,
    updated_at TEXT
);
CREATE INDEX IF NOT EXISTS idx_ss_jobs_plugin_status
    ON plugin_subtitlestudio_jobs (plugin_id, status, priority, queue_rank);
CREATE INDEX IF NOT EXISTS idx_ss_jobs_title
    ON plugin_subtitlestudio_jobs (plugin_id, title);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class JobStore:
    def __init__(self, data_path: Path, plugin_id: str = "SubtitleStudio"):
        self.plugin_id = plugin_id
        self.path = Path(data_path) / "jobs.sqlite"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._ensure()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _ensure(self) -> None:
        with self._lock, self._connect() as conn:
            conn.executescript(SCHEMA)

    def save_job(self, job: Job) -> Job:
        job.plugin_id = self.plugin_id
        job.updated_at = utc_now()
        if not job.created_at:
            job.created_at = job.updated_at
        payload = json.dumps(job.payload, ensure_ascii=False)
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO plugin_subtitlestudio_jobs (
                    job_id, plugin_id, title, path, media_source, media_id, tmdbid, doubanid,
                    priority, status, trigger, strategy, is_strm, error, created_at, updated_at,
                    started_at, finished_at, queue_rank, payload
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(job_id) DO UPDATE SET
                    title=excluded.title, path=excluded.path, media_source=excluded.media_source,
                    media_id=excluded.media_id, tmdbid=excluded.tmdbid, doubanid=excluded.doubanid,
                    priority=excluded.priority, status=excluded.status, trigger=excluded.trigger,
                    strategy=excluded.strategy, is_strm=excluded.is_strm, error=excluded.error,
                    updated_at=excluded.updated_at, started_at=excluded.started_at,
                    finished_at=excluded.finished_at, queue_rank=excluded.queue_rank,
                    payload=excluded.payload
                """,
                (
                    job.job_id, job.plugin_id, job.title, job.path, job.media_source, job.media_id,
                    job.tmdbid, job.doubanid, job.priority, job.status, job.trigger, job.strategy,
                    1 if job.is_strm else 0, job.error, job.created_at, job.updated_at,
                    job.started_at, job.finished_at, job.queue_rank, payload,
                ),
            )
        return job

    def get_job(self, job_id: str) -> Optional[Job]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM plugin_subtitlestudio_jobs WHERE job_id=? AND plugin_id=?",
                (job_id, self.plugin_id),
            ).fetchone()
        return self._job_from_row(row) if row else None

    def delete_job(self, job_id: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "DELETE FROM plugin_subtitlestudio_jobs WHERE job_id=? AND plugin_id=?",
                (job_id, self.plugin_id),
            )
            conn.execute(
                "DELETE FROM plugin_subtitlestudio_cues WHERE job_id=? AND plugin_id=?",
                (job_id, self.plugin_id),
            )

    def list_jobs(
        self,
        *,
        q: str = "",
        status: str = "",
        limit: int = 200,
    ) -> List[Job]:
        sql = "SELECT * FROM plugin_subtitlestudio_jobs WHERE plugin_id=?"
        args: List[Any] = [self.plugin_id]
        if status:
            sql += " AND status=?"
            args.append(status)
        if q:
            sql += " AND (title LIKE ? OR path LIKE ?)"
            like = f"%{q}%"
            args.extend([like, like])
        sql += " ORDER BY CASE status WHEN 'running' THEN 0 WHEN 'pending' THEN 1 ELSE 2 END, queue_rank ASC, created_at DESC"
        sql += " LIMIT ?"
        args.append(int(limit))
        with self._lock, self._connect() as conn:
            rows = conn.execute(sql, args).fetchall()
        return [self._job_from_row(row) for row in rows]

    def pending_jobs(self, *, allow_p2: bool = True) -> List[Job]:
        jobs = [item for item in self.list_jobs(status="pending") if item.status == "pending"]
        if not allow_p2:
            jobs = [item for item in jobs if item.priority != "P2"]
        return sorted(
            jobs,
            key=lambda item: (PRIORITY_ORDER.get(item.priority, 9), item.queue_rank, item.created_at),
        )

    def running_job(self) -> Optional[Job]:
        rows = self.list_jobs(status="running", limit=1)
        return rows[0] if rows else None

    def next_queue_rank(self) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT MIN(queue_rank) FROM plugin_subtitlestudio_jobs WHERE plugin_id=? AND status='pending'",
                (self.plugin_id,),
            ).fetchone()
        current = row[0] if row and row[0] is not None else 0
        return int(current) - 1

    def tail_queue_rank(self) -> int:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT MAX(queue_rank) FROM plugin_subtitlestudio_jobs WHERE plugin_id=? AND status='pending'",
                (self.plugin_id,),
            ).fetchone()
        current = row[0] if row and row[0] is not None else 0
        return int(current) + 1

    def save_graph(self, graph: CueGraph) -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO plugin_subtitlestudio_cues (job_id, plugin_id, graph, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(job_id) DO UPDATE SET graph=excluded.graph, updated_at=excluded.updated_at
                """,
                (graph.job_id, self.plugin_id, json.dumps(graph.to_dict(), ensure_ascii=False), utc_now()),
            )

    def get_graph(self, job_id: str) -> Optional[CueGraph]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT graph FROM plugin_subtitlestudio_cues WHERE job_id=? AND plugin_id=?",
                (job_id, self.plugin_id),
            ).fetchone()
        if not row or not row["graph"]:
            return None
        return CueGraph.from_dict(json.loads(row["graph"]))

    def counts(self) -> Dict[str, int]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM plugin_subtitlestudio_jobs WHERE plugin_id=? GROUP BY status",
                (self.plugin_id,),
            ).fetchall()
        data = {item["status"]: int(item["n"]) for item in rows}
        data["total"] = sum(data.values())
        return data

    def _job_from_row(self, row: sqlite3.Row) -> Job:
        payload = {}
        try:
            payload = json.loads(row["payload"] or "{}")
        except json.JSONDecodeError:
            payload = {}
        return Job.from_dict({
            "job_id": row["job_id"],
            "plugin_id": row["plugin_id"],
            "title": row["title"],
            "path": row["path"],
            "media_source": row["media_source"],
            "media_id": row["media_id"],
            "tmdbid": row["tmdbid"],
            "doubanid": row["doubanid"],
            "priority": row["priority"],
            "status": row["status"],
            "trigger": row["trigger"],
            "strategy": row["strategy"],
            "is_strm": bool(row["is_strm"]),
            "error": row["error"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "queue_rank": row["queue_rank"],
            "payload": payload,
        })
