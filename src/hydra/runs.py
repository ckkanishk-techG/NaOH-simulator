"""Run repository: persists every simulation with full provenance and supports bit-identical re-runs.

Each run stores parameters, seed, solver settings, code version (git hash), environment hash, property-database hash,
timestamp (``RunRecord``) in the database, and the result arrays in ``<data_dir>/runs/<run_id>.npz``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .core.l1 import SimResult, SolverSettings, simulate
from .core.params import ParamSet
from .core.scenario import Scenario
from .db import Database
from .provenance import RunRecord, make_record
from .thermo import propdb


class RunRepository:
    def __init__(self, db: Database, data_dir: Path) -> None:
        self.db, self.dir = db, Path(data_dir) / "runs"
        self.dir.mkdir(parents=True, exist_ok=True)
        db.execute("CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, created_utc TEXT, record TEXT, summary TEXT, kind TEXT, owner TEXT)")

    def record_for(self, sc: Scenario, overrides: dict[str, float], settings: SolverSettings, fidelity: str, seed: int = 0) -> RunRecord:
        solver = {"rtol": settings.rtol, "atol": settings.atol, "method": settings.method, "dt_out": settings.dt_out,
                  "max_step": settings.max_step, "property_db": propdb.db_hash()}
        return make_record({"scenario": json.loads(sc.model_dump_json()), "overrides": overrides}, seed, solver, fidelity)

    def run(self, sc: Scenario, overrides: dict[str, float] | None = None, settings: SolverSettings | None = None, owner: str = "local",
            seed: int = 0) -> tuple[RunRecord, SimResult]:
        settings = settings or SolverSettings()
        overrides = overrides or {}
        rec = self.record_for(sc, overrides, settings, sc.fidelity, seed)
        res = simulate(sc, ParamSet(overrides), settings)
        self.save(rec, res, owner)
        return rec, res

    def save(self, rec: RunRecord, res: SimResult, owner: str = "local") -> None:
        arrays = {f"s_{k}": np.asarray(v) for k, v in res.series.items()}
        arrays["t"] = res.t
        kw: dict[str, Any] = dict(arrays)
        np.savez_compressed(self.dir / f"{rec.run_id}.npz", **kw)
        self.db.execute("DELETE FROM runs WHERE run_id=?", (rec.run_id,))
        self.db.execute("INSERT INTO runs (run_id, created_utc, record, summary, kind, owner) VALUES (?,?,?,?,?,?)",
                        (rec.run_id, rec.created_utc, rec.model_dump_json(),
                         json.dumps({"summary": res.summary, "ledger": res.ledger, "diag": res.diagnostics,
                                     "events": res.events}, default=float), rec.fidelity, owner))

    def record(self, run_id: str) -> RunRecord:
        row = self.db.one("SELECT record FROM runs WHERE run_id=?", (run_id,))
        if row is None:
            raise KeyError(run_id)
        return RunRecord.model_validate_json(row["record"])

    def summary(self, run_id: str) -> dict[str, Any]:
        row = self.db.one("SELECT summary FROM runs WHERE run_id=?", (run_id,))
        if row is None:
            raise KeyError(run_id)
        return json.loads(row["summary"])  # type: ignore[no-any-return]

    def series(self, run_id: str) -> dict[str, np.ndarray]:
        z = np.load(self.dir / f"{run_id}.npz")
        out = {k[2:]: z[k] for k in z.files if k.startswith("s_")}
        out["t"] = z["t"]
        return out

    def list(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.db.query("SELECT run_id, created_utc, kind, owner FROM runs ORDER BY created_utc DESC LIMIT ?", (limit,))

    def rerun(self, run_id: str) -> tuple[bool, SimResult]:
        """Re-run from the stored record and compare arrays BIT-IDENTICALLY with the stored result."""
        rec = self.record(run_id)
        sc = Scenario.model_validate(rec.parameters["scenario"])
        s = rec.solver
        settings = SolverSettings(rtol=s["rtol"], atol=s["atol"], method=s["method"], dt_out=s["dt_out"], max_step=s["max_step"])
        res = simulate(sc, ParamSet(rec.parameters["overrides"]), settings)
        old = self.series(run_id)
        same = np.array_equal(old["t"], res.t) and all(np.array_equal(old[k], res.series[k], equal_nan=True) for k in res.series)
        return bool(same), res
