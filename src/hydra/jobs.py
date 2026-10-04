"""Job queue for long runs: in-process thread pool (laptop) or Redis/RQ (server), with per-job CPU/RAM limits.

Jobs are rows in the ``jobs`` table (status, params, result, error, progress). Handlers are registered by ``kind``
and receive ``(params, ctx)``; ``ctx.progress(frac, msg)`` reports progress and ``ctx.cancelled()`` supports
cooperative cancellation. ``run_limited`` executes a callable in a child process with ``RLIMIT_AS`` (RAM) and CPU
affinity limits - the mechanism behind the configurable server-mode limits."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .db import Database

Handler = Callable[[dict[str, Any], "JobContext"], dict[str, Any]]
HANDLERS: dict[str, Handler] = {}


def register(kind: str) -> Callable[[Handler], Handler]:
    def deco(fn: Handler) -> Handler:
        HANDLERS[kind] = fn
        return fn

    return deco


class JobCancelled(Exception):
    pass


@dataclass
class JobContext:
    job_id: str
    queue: JobQueue

    def progress(self, frac: float, msg: str = "") -> None:
        self.queue._set(self.job_id, progress=float(min(max(frac, 0.0), 1.0)), message=msg)
        if self.cancelled():
            raise JobCancelled()

    def cancelled(self) -> bool:
        row = self.queue.db.one("SELECT status FROM jobs WHERE id=?", (self.job_id,))
        return bool(row and row["status"] == "cancelling")


def _child(fn: Callable[..., Any], args: tuple[Any, ...], kwargs: dict[str, Any], ram_mb: int | None, cpus: int | None, q: Any) -> None:
    try:
        if ram_mb:
            import resource

            lim = ram_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (lim, lim))
        if cpus and hasattr(os, "sched_setaffinity"):
            allowed = sorted(os.sched_getaffinity(0))[:cpus]
            os.sched_setaffinity(0, set(allowed))
        q.put(("ok", fn(*args, **kwargs)))
    except MemoryError:
        q.put(("err", "MemoryError: job exceeded its RAM limit"))
    except BaseException as exc:  # noqa: BLE001 - report everything to the parent
        q.put(("err", f"{type(exc).__name__}: {exc}"))


def run_limited(fn: Callable[..., Any], *args: Any, ram_mb: int | None = None, cpus: int | None = None, timeout_s: float | None = None,
                **kwargs: Any) -> Any:
    """Run ``fn`` in a forked child with RAM (address-space) and CPU-count limits and an optional wall-clock timeout."""
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_child, args=(fn, args, kwargs, ram_mb, cpus, q))
    p.start()
    try:
        status, val = q.get(timeout=timeout_s)
    except Exception as exc:
        p.terminate()
        p.join()
        raise TimeoutError(f"job exceeded {timeout_s}s") from exc
    p.join()
    if status == "err":
        raise RuntimeError(val)
    return val


class JobQueue:
    def __init__(self, db: Database, workers: int = 2, max_cpus: int | None = None, max_ram_mb: int | None = None,
                 isolate: bool = False) -> None:
        self.db = db
        self.max_cpus, self.max_ram_mb, self.isolate = max_cpus, max_ram_mb, isolate
        self.pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="hydra-job")
        self.futures: dict[str, Future[Any]] = {}
        db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, kind TEXT, owner TEXT, status TEXT, params TEXT, result TEXT, "
                   "error TEXT, progress REAL, message TEXT, created REAL, started REAL, finished REAL, cpus INTEGER, ram_mb INTEGER)")

    def _set(self, job_id: str, **kw: Any) -> None:
        cols = ", ".join(f"{k}=?" for k in kw)
        self.db.execute(f"UPDATE jobs SET {cols} WHERE id=?", (*kw.values(), job_id))  # noqa: S608 - column names are internal literals

    def submit(self, kind: str, params: dict[str, Any], owner: str = "local", cpus: int | None = None, ram_mb: int | None = None) -> str:
        if kind not in HANDLERS:
            raise KeyError(f"unknown job kind {kind!r}; available: {sorted(HANDLERS)}")
        cpus = min(cpus or self.max_cpus or 1, self.max_cpus or 1 << 30)
        ram = min(ram_mb or self.max_ram_mb or 1 << 30, self.max_ram_mb or 1 << 30) if (ram_mb or self.max_ram_mb) else None
        jid = uuid.uuid4().hex[:12]
        self.db.execute("INSERT INTO jobs (id, kind, owner, status, params, progress, message, created, cpus, ram_mb) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (jid, kind, owner, "queued", json.dumps(params), 0.0, "", time.time(), cpus, ram))
        self.futures[jid] = self.pool.submit(self._run, jid)
        return jid

    def _run(self, jid: str) -> None:
        row = self.db.one("SELECT kind, params, cpus, ram_mb FROM jobs WHERE id=?", (jid,))
        assert row is not None
        if self.db.one("SELECT status FROM jobs WHERE id=?", (jid,))["status"] == "cancelling":  # type: ignore[index]
            self._set(jid, status="cancelled", finished=time.time())
            return
        self._set(jid, status="running", started=time.time())
        try:
            fn = HANDLERS[row["kind"]]
            params = json.loads(row["params"])
            if self.isolate:
                res = run_limited(fn, params, JobContext(jid, self), ram_mb=row["ram_mb"], cpus=row["cpus"])
            else:
                res = fn(params, JobContext(jid, self))
            self._set(jid, status="done", result=json.dumps(res, default=_json_default), progress=1.0, finished=time.time())
        except JobCancelled:
            self._set(jid, status="cancelled", finished=time.time())
        except Exception as exc:  # noqa: BLE001
            self._set(jid, status="failed", error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()[-1500:]}", finished=time.time())

    def get(self, jid: str) -> dict[str, Any] | None:
        row = self.db.one("SELECT * FROM jobs WHERE id=?", (jid,))
        if row is None:
            return None
        for k in ("params", "result"):
            if row.get(k):
                row[k] = json.loads(row[k])
        return row

    def cancel(self, jid: str) -> bool:
        row = self.db.one("SELECT status FROM jobs WHERE id=?", (jid,))
        if row is None or row["status"] in ("done", "failed", "cancelled"):
            return False
        self._set(jid, status="cancelling")
        fut = self.futures.get(jid)
        if fut is not None and fut.cancel():
            self._set(jid, status="cancelled", finished=time.time())
        return True

    def wait(self, jid: str, timeout: float = 120.0) -> dict[str, Any]:
        fut = self.futures.get(jid)
        if fut is not None:
            fut.result(timeout=timeout)
        out = self.get(jid)
        assert out is not None
        return out

    def list(self, owner: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        if owner:
            return self.db.query("SELECT id, kind, owner, status, progress, created, finished FROM jobs WHERE owner=? ORDER BY created DESC LIMIT ?", (owner, limit))
        return self.db.query("SELECT id, kind, owner, status, progress, created, finished FROM jobs ORDER BY created DESC LIMIT ?", (limit,))

    def shutdown(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)


class RQQueue:
    """Server-mode adapter: enqueue the job id on a Redis/RQ queue; ``hydra-worker`` processes execute it with
    :func:`run_job_by_id`. Requires ``redis`` and ``rq`` (server extra)."""

    def __init__(self, db: Database, redis_url: str, queue_name: str = "hydra") -> None:
        import redis
        from rq import Queue

        self.db = db
        self.q = Queue(queue_name, connection=redis.from_url(redis_url))
        JobQueue(db, workers=1)  # creates the table

    def submit(self, kind: str, params: dict[str, Any], owner: str = "local", cpus: int | None = None, ram_mb: int | None = None) -> str:
        jid = uuid.uuid4().hex[:12]
        self.db.execute("INSERT INTO jobs (id, kind, owner, status, params, progress, message, created, cpus, ram_mb) VALUES (?,?,?,?,?,?,?,?,?,?)",
                        (jid, kind, owner, "queued", json.dumps(params), 0.0, "", time.time(), cpus, ram_mb))
        self.q.enqueue("hydra.jobs.run_job_by_id", self.db.url, jid, job_timeout=24 * 3600)  # magic: 24 h
        return jid


def run_job_by_id(db_url: str, jid: str) -> None:
    """RQ worker entry point: run a queued job row to completion (respecting its RAM/CPU limits)."""
    q = JobQueue(Database(db_url), workers=1, isolate=True)
    q._run(jid)


def _json_default(o: Any) -> Any:
    import numpy as np

    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    raise TypeError(f"not JSON serialisable: {type(o)}")


_ = threading
