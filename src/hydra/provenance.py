"""Run provenance: every run records parameters, seed, solver settings, code version,
environment hash and timestamp, so any result can be re-run bit-identically.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sqlite3
import subprocess
import sys
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from . import __version__


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(obj: Any) -> str:
    return hashlib.sha256(canonical_json(obj).encode()).hexdigest()


def git_hash(cwd: Path | None = None) -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def environment_info() -> dict[str, Any]:
    pkgs = {}
    for name in ("numpy", "scipy", "pydantic", "fastapi", "pint"):
        try:
            pkgs[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pkgs[name] = "absent"
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": pkgs}


class RunRecord(BaseModel):
    run_id: str
    created_utc: str
    hydra_version: str
    git_hash: str
    env_hash: str
    seed: int
    fidelity: str
    parameters: dict[str, Any]
    solver: dict[str, Any]


def make_record(
    parameters: dict[str, Any],
    seed: int,
    solver: dict[str, Any] | None = None,
    fidelity: str = "L1",
    now: datetime | None = None,
) -> RunRecord:
    """Build a record. ``run_id`` is a content hash, so identical inputs give identical ids."""
    solver = solver or {}
    env = environment_info()
    head = {
        "version": __version__,
        "git": git_hash(),
        "env": sha256(env),
        "seed": seed,
        "fidelity": fidelity,
        "parameters": parameters,
        "solver": solver,
    }
    return RunRecord(
        run_id=sha256(head)[:16],
        created_utc=(now or datetime.now(UTC)).isoformat(),
        hydra_version=__version__,
        git_hash=head["git"],
        env_hash=head["env"],
        seed=seed,
        fidelity=fidelity,
        parameters=parameters,
        solver=solver,
    )


class RunStore:
    """SQLite run store (laptop default). PostgreSQL arrives with server mode (stage 16)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(path)
        self.con.execute(
            "CREATE TABLE IF NOT EXISTS runs "
            "(run_id TEXT PRIMARY KEY, created_utc TEXT, record TEXT)"
        )

    def save(self, rec: RunRecord) -> None:
        with self.con:
            self.con.execute(
                "INSERT OR REPLACE INTO runs VALUES (?,?,?)",
                (rec.run_id, rec.created_utc, rec.model_dump_json()),
            )

    def load(self, run_id: str) -> RunRecord:
        row = self.con.execute("SELECT record FROM runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(run_id)
        return RunRecord.model_validate_json(row[0])

    def list_ids(self) -> list[str]:
        return [r[0] for r in self.con.execute("SELECT run_id FROM runs ORDER BY created_utc")]
