"""Local electronic lab notebook: experiments, attachments (content-addressed files), provenance graph, calibration records.

Everything lives in the local database and ``<data_dir>/eln/files``; nothing leaves the machine."""

from __future__ import annotations

import builtins
import hashlib
import json
import re
import shutil
import time
import uuid
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np

from ..constants import KELVIN_OFFSET, V_STP_L
from ..core.params import ParamSet
from ..core.scenario import Scenario
from ..db import Database
from ..hardware.sources import parse_line
from ..inference.data import Channel, Experiment
from ..inference.sensors import Sensor

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
NODE_TYPES = ("experiment", "attachment", "twin_run", "calibration", "validation", "report", "blind_prediction", "parameter_set")


def safe_filename(name: str) -> str:
    base = Path(name).name  # strips any directory components (path traversal)
    return SAFE_NAME.sub("_", base).strip("._") or "file"


class ELN:
    def __init__(self, db: Database, data_dir: Path) -> None:
        self.db, self.files = db, Path(data_dir) / "eln" / "files"
        self.files.mkdir(parents=True, exist_ok=True)
        db.execute("CREATE TABLE IF NOT EXISTS eln_experiments (id TEXT PRIMARY KEY, title TEXT, created REAL, operator TEXT, status TEXT, "
                   "setup TEXT, quantities TEXT, conditions TEXT, notes TEXT, batch TEXT, split TEXT)")
        db.execute("CREATE TABLE IF NOT EXISTS eln_attachments (id TEXT PRIMARY KEY, exp_id TEXT, filename TEXT, sha256 TEXT, size INTEGER, "
                   "kind TEXT, path TEXT, created REAL)")
        db.execute("CREATE TABLE IF NOT EXISTS eln_links (src_type TEXT, src_id TEXT, dst_type TEXT, dst_id TEXT, relation TEXT, created REAL)")
        db.execute("CREATE TABLE IF NOT EXISTS eln_calibrations (id TEXT PRIMARY KEY, created REAL, record TEXT)")

    # --- experiments ---------------------------------------------------------------------------------------
    def create_experiment(self, title: str, operator: str = "local", setup: dict[str, Any] | None = None, quantities: dict[str, Any] | None = None,
                          conditions: dict[str, Any] | None = None, notes: str = "", batch: str = "A", split: str = "calib") -> str:
        if split not in ("calib", "val", "blind"):
            raise ValueError("split must be calib/val/blind")
        if setup:
            Scenario(**setup)  # validate
        eid = "E" + uuid.uuid4().hex[:8]
        self.db.execute("INSERT INTO eln_experiments VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                        (eid, title, time.time(), operator, "planned", json.dumps(setup or {}), json.dumps(quantities or {}),
                         json.dumps(conditions or {}), notes, batch, split))
        return eid

    def get(self, eid: str) -> dict[str, Any]:
        row = self.db.one("SELECT * FROM eln_experiments WHERE id=?", (eid,))
        if row is None:
            raise KeyError(eid)
        for k in ("setup", "quantities", "conditions"):
            row[k] = json.loads(row[k] or "{}")
        row["attachments"] = self.db.query("SELECT id, filename, sha256, size, kind FROM eln_attachments WHERE exp_id=?", (eid,))
        return row

    def list_experiments(self) -> builtins.list[dict[str, Any]]:
        return self.db.query("SELECT id, title, created, operator, status, batch, split FROM eln_experiments ORDER BY created DESC")

    def update(self, eid: str, **fields: Any) -> None:
        allowed = {"title", "status", "notes", "batch", "split", "operator"}
        bad = set(fields) - allowed - {"setup", "quantities", "conditions"}
        if bad:
            raise ValueError(f"cannot update {bad}")
        for k, v in fields.items():
            self.db.execute(f"UPDATE eln_experiments SET {k}=? WHERE id=?", (json.dumps(v) if isinstance(v, dict) else v, eid))  # noqa: S608 - whitelisted names

    # --- attachments (content addressed) ------------------------------------------------------------------------
    def add_attachment(self, eid: str, filename: str, data: bytes, kind: str = "other") -> dict[str, Any]:
        if kind not in ("photo", "sensor", "notes", "other"):
            raise ValueError("kind must be photo/sensor/notes/other")
        self.get(eid)
        digest = hashlib.sha256(data).hexdigest()
        dest = self.files / digest[:2] / digest
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(data)
        aid = "A" + uuid.uuid4().hex[:8]
        name = safe_filename(filename)
        self.db.execute("INSERT INTO eln_attachments VALUES (?,?,?,?,?,?,?,?)", (aid, eid, name, digest, len(data), kind, str(dest), time.time()))
        self.link("experiment", eid, "attachment", aid, "attached")
        return {"id": aid, "sha256": digest, "filename": name, "size": len(data)}

    def read_attachment(self, aid: str) -> tuple[str, bytes]:
        row = self.db.one("SELECT filename, path, sha256 FROM eln_attachments WHERE id=?", (aid,))
        if row is None:
            raise KeyError(aid)
        data = Path(row["path"]).read_bytes()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("attachment failed its integrity check")
        return row["filename"], data

    # --- provenance graph ---------------------------------------------------------------------------------------------
    def link(self, src_type: str, src_id: str, dst_type: str, dst_id: str, relation: str) -> None:
        for t in (src_type, dst_type):
            if t not in NODE_TYPES:
                raise ValueError(f"unknown node type {t!r}")
        self.db.execute("INSERT INTO eln_links VALUES (?,?,?,?,?,?)", (src_type, src_id, dst_type, dst_id, relation, time.time()))

    def link_twin_run(self, eid: str, run_id: str) -> None:
        self.link("experiment", eid, "twin_run", run_id, "simulated_by")

    def record_calibration(self, exp_ids: list[str], theta: dict[str, float], intervals: dict[str, Any], meta: dict[str, Any] | None = None) -> str:
        cid = "C" + uuid.uuid4().hex[:8]
        rec = {"theta": theta, "intervals": intervals, "experiments": exp_ids, "meta": meta or {}}
        self.db.execute("INSERT INTO eln_calibrations VALUES (?,?,?)", (cid, time.time(), json.dumps(rec, default=float)))
        for e in exp_ids:
            self.link("experiment", e, "calibration", cid, "calibrated_with")
        return cid

    def provenance_graph(self, node_type: str, node_id: str) -> dict[str, Any]:
        """All nodes/edges reachable (in either direction) from a node: the full provenance of a result."""
        seen = {(node_type, node_id)}
        edges: list[dict[str, Any]] = []
        dq = deque([(node_type, node_id)])
        while dq:
            t, i = dq.popleft()
            rows = self.db.query("SELECT * FROM eln_links WHERE (src_type=? AND src_id=?) OR (dst_type=? AND dst_id=?)", (t, i, t, i))
            for r in rows:
                edge = {"source": f"{r['src_type']}:{r['src_id']}", "target": f"{r['dst_type']}:{r['dst_id']}", "relation": r["relation"]}
                if edge not in edges:
                    edges.append(edge)
                for nt, nid in ((r["src_type"], r["src_id"]), (r["dst_type"], r["dst_id"])):
                    if (nt, nid) not in seen:
                        seen.add((nt, nid))
                        dq.append((nt, nid))
        return {"nodes": sorted(f"{t}:{i}" for t, i in seen), "edges": edges}

    # --- from notebook to calibration data -------------------------------------------------------------------------------
    def to_experiment(self, eid: str, n_cells: int = 4) -> Experiment:
        """Build a calibration ``Experiment`` from the logged sensor CSV (frame format of ``hydra.hardware.sources``).

        Channels: liquid temperature (T1) with a thermocouple lag model, wall temperature (T2), and cumulative H2 moles
        from integrating the logged STP flow."""
        e = self.get(eid)
        sensors = [a for a in e["attachments"] if a["kind"] == "sensor"]
        if not sensors:
            raise ValueError("experiment has no sensor attachment")
        _, data = self.read_attachment(sensors[0]["id"])
        frames = [f for f in (parse_line(ln) for ln in data.decode(errors="ignore").splitlines()) if f is not None]
        if len(frames) < 3:
            raise ValueError("sensor file has fewer than 3 valid frames")
        t = np.array([f.t for f in frames])
        sc = Scenario(**{**e["setup"], "duration_s": float(t[-1])})
        chans: list[Channel] = []
        t1 = np.array([f.T1 for f in frames]) + KELVIN_OFFSET
        ok = np.isfinite(t1)
        if ok.sum() > 2:
            chans.append(Channel("T", t[ok], t1[ok], np.full(ok.sum(), 0.2), Sensor("tc", "K", tau_s=8.0, noise_sd=0.15)))
        t2 = np.array([f.T2 for f in frames]) + KELVIN_OFFSET
        ok2 = np.isfinite(t2)
        if ok2.sum() > 2:
            chans.append(Channel("Tw", t[ok2], t2[ok2], np.full(ok2.sum(), 0.3), Sensor("wall", "K")))
        fl = np.array([f.flow for f in frames])
        if np.isfinite(fl).sum() > 2:
            flow_l_s = np.nan_to_num(fl, nan=0.0) / 60.0
            cum_l = np.concatenate([[0.0], np.cumsum(0.5 * (flow_l_s[1:] + flow_l_s[:-1]) * np.diff(t))])
            n_mol = cum_l / V_STP_L
            chans.append(Channel("n_h2", t, n_mol, np.maximum(3e-4, 0.02 * n_mol)))
        if not chans:
            raise ValueError("no usable channels in the sensor file")
        _ = n_cells
        return Experiment(eid, sc, chans, e["batch"], e["split"], 1.0, {"title": e["title"]})

    # --- bundles ------------------------------------------------------------------------------------------------------------
    def export_bundle(self, eid: str, dest: Path) -> Path:
        """Experiment metadata + attachments copied into a folder (portable record)."""
        e = self.get(eid)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "experiment.json").write_text(json.dumps(e, indent=1, default=str))
        for a in e["attachments"]:
            name, data = self.read_attachment(a["id"])
            (dest / f"{a['id']}_{name}").write_bytes(data)
        (dest / "provenance.json").write_text(json.dumps(self.provenance_graph("experiment", eid), indent=1))
        return dest


_ = (ParamSet, shutil)
