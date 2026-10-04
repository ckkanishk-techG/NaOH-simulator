r"""Validation protocol (spec 20.6): strict calibration / validation / sealed-blind separation, locked blind
predictions stored with a timestamp and hash, metrics incl. uncertainty calibration (reliability), extrapolation
tests and an auto-generated validation report.

Never claim accuracy that has not been measured on held-out data: ``accuracy_claim`` only reads blind/validation
scores computed by :func:`score`.
"""

from __future__ import annotations

import hashlib
import html
import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import norm

from ..core.params import ParamSet
from ..provenance import canonical_json
from .calibrate import Predictor
from .data import Experiment


# ---------------------------------------------------------------- splits
def split_experiments(exps: list[Experiment]) -> dict[str, list[Experiment]]:
    out: dict[str, list[Experiment]] = {"calib": [], "val": [], "blind": []}
    for e in exps:
        out[e.split].append(e)
    return out


def assert_no_leakage(calib: list[Experiment], held: list[Experiment]) -> None:
    ids = {e.id for e in calib}
    clash = [e.id for e in held if e.id in ids]
    if clash:
        raise ValueError(f"held-out experiments also in calibration set: {clash}")
    for e in calib:
        if e.split == "blind":
            raise ValueError(f"sealed blind experiment {e.id} must not be used for calibration")


# ---------------------------------------------------------------- metrics
def interval_from_samples(draws: np.ndarray, level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
    a = (1 - level) / 2
    return np.quantile(draws, a, axis=0), np.quantile(draws, 1 - a, axis=0)


def point_metrics(t: np.ndarray, y: np.ndarray, pred: np.ndarray, kind: str = "n_h2") -> dict[str, float]:
    err = pred - y
    m: dict[str, float] = {"rmse": float(np.sqrt(np.mean(err**2))), "max_abs_err": float(np.max(np.abs(err))),
                           "bias": float(np.mean(err)), "n": float(len(y))}
    nz = np.abs(y) > 1e-9 * max(np.max(np.abs(y)), 1e-300)  # magic: avoid dividing by ~0
    m["mape"] = float(100.0 * np.mean(np.abs(err[nz] / y[nz]))) if nz.any() else float("nan")
    if kind == "n_h2" and y[-1] > 0:
        def t90(a: np.ndarray) -> float:
            tot = a[-1]
            return float(np.interp(0.9 * tot, np.maximum.accumulate(a), t))  # magic: 90 %

        m["t90_err_s"] = t90(pred) - t90(y)
    if kind in ("T", "Tw"):
        m["peak_err"] = float(pred.max() - y.max())
    return m


def reliability(y: np.ndarray, mean: np.ndarray, sd: np.ndarray, levels: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Empirical coverage of central predictive intervals vs nominal level (Gaussian predictive distribution)."""
    levels = np.linspace(0.1, 0.99, 10) if levels is None else levels  # magic: reliability grid
    z = np.abs(y - mean) / np.maximum(sd, 1e-300)  # magic: floor
    cov = np.array([np.mean(z <= norm.ppf(0.5 + lv / 2)) for lv in levels])
    return {"nominal": levels, "empirical": cov}


def calibration_verdict(rel: dict[str, np.ndarray], tol: float = 0.1) -> str:
    """'overconfident' if empirical coverage is below nominal by more than ``tol`` on average (a failing model)."""
    gap = float(np.mean(rel["empirical"] - rel["nominal"]))
    if gap < -tol:
        return "overconfident"
    if gap > tol:
        return "underconfident"
    return "calibrated"


def score(pred: Predictor, params: ParamSet, exps: list[Experiment], band_sd: dict[str, list[list[np.ndarray]]] | None = None,
          model_err_rel: float = 0.0) -> dict[str, Any]:
    """Held-out scoring of a locked parameter set: per-experiment/channel metrics + pooled reliability.

    ``band_sd`` (optional): predictive sd per experiment id/channel (parameter + noise); otherwise the
    measurement sigma plus ``model_err_rel`` is used (which flags overconfidence if too small)."""
    rows = []
    ys, ms, ss = [], [], []
    for e in exps:
        sim = pred.sim(e, params)
        for ic, c in enumerate(e.channels):
            pr = pred.channel_prediction(c, sim)
            sd = np.sqrt(c.sigma**2 + (model_err_rel * pr) ** 2) if band_sd is None else band_sd[e.id][ic]
            rows.append({"exp": e.id, "kind": c.kind, **point_metrics(c.t, c.y, pr, c.kind), "batch": e.batch})
            ys.append(c.y)
            ms.append(pr)
            ss.append(sd)
    y, m, s = np.concatenate(ys), np.concatenate(ms), np.concatenate(ss)
    rel = reliability(y, m, s)
    return {"rows": rows, "reliability": rel, "verdict": calibration_verdict(rel),
            "coverage95": float(np.mean(np.abs(y - m) <= 1.96 * s)),  # magic: 95 % normal quantile
            "rmse_all": float(np.sqrt(np.mean((y - m) ** 2)))}


def extrapolation_test(pred_factory: Any, exps: list[Experiment], key: str, narrow: tuple[float, float],
                       names: list[str], fit_fn: Any) -> dict[str, Any]:
    """Calibrate on experiments whose ``key`` (scenario field) lies in ``narrow``, predict all others and report
    RMSE versus distance outside the calibrated range (how fast accuracy degrades)."""
    inside = [e for e in exps if narrow[0] <= getattr(e.scenario, key) <= narrow[1]]
    outside = [e for e in exps if e not in inside]
    pred = pred_factory(inside)
    fit = fit_fn(pred, names)
    rows = []
    for e in outside:
        p2 = pred_factory([e])
        r = p2.residuals(p2.base.with_(**fit.theta))
        v = getattr(e.scenario, key)
        dist = max(narrow[0] - v, v - narrow[1])
        rows.append({"exp": e.id, "distance": float(dist), "rms_norm_resid": float(np.sqrt(np.mean(r**2)))})
    return {"fit": fit, "rows": sorted(rows, key=lambda r: r["distance"]), "n_calibration": len(inside)}


# ---------------------------------------------------------------- blind prediction (locked before the experiment)
def model_hash(params: ParamSet, code_hash: str = "", extra: dict[str, Any] | None = None) -> str:
    return hashlib.sha256(canonical_json({"p": params.v, "code": code_hash, "x": extra or {}}).encode()).hexdigest()


class BlindStore:
    """SQLite store of locked predictions. A prediction is immutable once stored (hash + UTC timestamp)."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(path)
        self.con.execute("CREATE TABLE IF NOT EXISTS blind (id TEXT PRIMARY KEY, created TEXT, model_hash TEXT, "
                         "scenario TEXT, payload TEXT, digest TEXT, scored TEXT)")

    def lock(self, exp_id: str, scenario: Any, params: ParamSet, t: np.ndarray, mean: np.ndarray, sd: np.ndarray,
             kind: str = "n_h2", code_hash: str = "") -> str:
        payload = {"kind": kind, "t": t.tolist(), "mean": mean.tolist(), "sd": sd.tolist()}
        mh = model_hash(params, code_hash)
        sc_json = scenario.model_dump_json()
        digest = hashlib.sha256((mh + sc_json + canonical_json(payload)).encode()).hexdigest()
        try:
            with self.con:
                self.con.execute("INSERT INTO blind VALUES (?,?,?,?,?,?,NULL)",
                                 (exp_id, datetime.now(UTC).isoformat(), mh, sc_json, json.dumps(payload), digest))
        except sqlite3.IntegrityError as exc:
            raise ValueError(f"prediction for {exp_id} is already locked and cannot be replaced") from exc
        return digest

    def verify(self, exp_id: str) -> bool:
        row = self.con.execute("SELECT model_hash, scenario, payload, digest FROM blind WHERE id=?", (exp_id,)).fetchone()
        if row is None:
            raise KeyError(exp_id)
        mh, sc, payload, digest = row
        return hashlib.sha256((mh + sc + canonical_json(json.loads(payload))).encode()).hexdigest() == digest

    def score(self, exp_id: str, t_obs: np.ndarray, y_obs: np.ndarray, sigma_obs: np.ndarray) -> dict[str, Any]:
        if not self.verify(exp_id):
            raise ValueError("stored prediction failed its integrity check")
        row = self.con.execute("SELECT payload FROM blind WHERE id=?", (exp_id,)).fetchone()
        p = json.loads(row[0])
        mean = np.interp(t_obs, p["t"], p["mean"])
        sd = np.sqrt(np.interp(t_obs, p["t"], p["sd"]) ** 2 + sigma_obs**2)
        out = {"metrics": point_metrics(t_obs, y_obs, mean, p["kind"]), "coverage95": float(np.mean(np.abs(y_obs - mean) <= 1.96 * sd)),  # magic: normal quantile
               "reliability": reliability(y_obs, mean, sd)}
        with self.con:
            self.con.execute("UPDATE blind SET scored=? WHERE id=?", (json.dumps({"metrics": out["metrics"],
                             "coverage95": out["coverage95"]}), exp_id))
        return out


def accuracy_claim(scores: dict[str, Any] | None) -> str:
    """The only sanctioned accuracy statement: it exists only if a held-out score exists."""
    if not scores:
        return "No accuracy claim: no held-out (validation/blind) data has been scored."
    return (f"Held-out RMSE {scores['rmse_all']:.3g} (n experiments={len({r['exp'] for r in scores['rows']})}), 95 % band "
            f"coverage {100 * scores['coverage95']:.0f}% -> uncertainty is {scores['verdict']}.")


def validation_report(title: str, scores: dict[str, Any] | None, extra: dict[str, Any] | None = None, path: Path | None = None) -> str:
    """Self-contained HTML validation report (offline)."""
    def fmt(v: Any) -> str:
        return html.escape(f"{v:.4g}" if isinstance(v, float) else str(v))

    rows = ""
    if scores:
        for r in scores["rows"]:
            rows += "<tr>" + "".join(f"<td>{fmt(r.get(k, ''))}</td>" for k in
                                     ("exp", "kind", "rmse", "mape", "max_abs_err", "bias", "t90_err_s", "peak_err")) + "</tr>"
    rel = ""
    if scores:
        rel = "".join(f"<tr><td>{a:.2f}</td><td>{b:.2f}</td></tr>" for a, b in
                      zip(scores["reliability"]["nominal"], scores["reliability"]["empirical"], strict=True))
    body = f"""<html><head><meta charset='utf-8'><title>{html.escape(title)}</title>
<style>body{{font:14px sans-serif;max-width:900px;margin:2em auto}}td,th{{border:1px solid #888;padding:2px 8px}}</style></head><body>
<h1>{html.escape(title)}</h1><p><b>{html.escape(accuracy_claim(scores))}</b></p>
<h2>Per-experiment metrics</h2><table><tr><th>exp</th><th>kind</th><th>RMSE</th><th>MAPE %</th><th>max err</th><th>bias</th><th>t90 err s</th><th>peak err</th></tr>{rows}</table>
<h2>Reliability (nominal vs empirical coverage)</h2><table><tr><th>nominal</th><th>empirical</th></tr>{rel}</table>
<p>Verdict: {html.escape(scores['verdict']) if scores else 'n/a'}. Overconfident models are flagged as failing.</p>
<pre>{html.escape(json.dumps(extra or {}, indent=1, default=str)[:4000])}</pre></body></html>"""
    if path:
        Path(path).write_text(body)
    return body
