"""Literature benchmark suite: dataset loader, plot digitiser, leaderboard and accuracy-regression gate."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .. import __version__
from ..core import l1_fast as lf
from ..core.l1 import SolverSettings, simulate
from ..core.params import ParamSet
from ..core.scenario import Scenario
from ..provenance import git_hash
from .data import Channel, Experiment

REPO_BENCH = Path(__file__).resolve().parents[3] / "benchmarks"
REQUIRED = ("id", "citation", "conditions", "t_s", "h2_mol", "digitization_sd_mol")


def load_dataset(path: Path) -> dict[str, Any]:
    d = json.loads(Path(path).read_text())
    missing = [k for k in REQUIRED if k not in d]
    if missing:
        raise ValueError(f"{path.name}: missing {missing}")
    if len(d["citation"]) < 10:  # magic: a real citation is mandatory
        raise ValueError(f"{path.name}: a real citation is required")
    if len(d["t_s"]) != len(d["h2_mol"]):
        raise ValueError(f"{path.name}: t_s and h2_mol length differ")
    return d


def load_all(folder: Path | None = None) -> list[dict[str, Any]]:
    folder = folder or REPO_BENCH / "datasets"
    return [load_dataset(p) for p in sorted(folder.glob("*.json"))]


def to_experiment(d: dict[str, Any]) -> Experiment:
    sc = Scenario(mode="open", duration_s=float(max(d["t_s"])), **d["conditions"])
    sd = float(d["digitization_sd_mol"])
    ch = Channel("n_h2", np.array(d["t_s"], float), np.array(d["h2_mol"], float), np.full(len(d["t_s"]), sd))
    return Experiment(d["id"], sc, [ch], meta={"citation": d["citation"]})


# ---------------------------------------------------------------- plot digitiser
def calibrate_axes(px: list[tuple[float, float]], vals: list[tuple[float, float]], log_x: bool = False, log_y: bool = False
                   ) -> Any:
    """Return f(pixel_x, pixel_y) -> (x, y) from two reference points per axis.

    ``px`` = [(x_px_1, y_px_1), (x_px_2, y_px_2)] pixel positions of two known axis points; ``vals`` the data
    values there as [(x1, y1), (x2, y2)] (x from the x-axis ticks, y from the y-axis ticks)."""
    (px1, py1), (px2, py2) = px
    (x1, y1), (x2, y2) = vals
    fx = (lambda v: np.log10(v)) if log_x else (lambda v: v)
    fy = (lambda v: np.log10(v)) if log_y else (lambda v: v)
    sx = (fx(x2) - fx(x1)) / (px2 - px1)
    sy = (fy(y2) - fy(y1)) / (py2 - py1)

    def convert(ppx: np.ndarray, ppy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = fx(x1) + (np.asarray(ppx, float) - px1) * sx
        y = fy(y1) + (np.asarray(ppy, float) - py1) * sy
        return (10.0**x if log_x else x), (10.0**y if log_y else y)

    return convert


def digitize_image(image_path: Path, color: tuple[int, int, int], tol: float = 60.0, min_pixels: int = 1) -> np.ndarray:
    """Pixel centroids (x_px, y_px) of the curve/marker colour, one point per image column that has matching pixels.

    Use together with :func:`calibrate_axes`. Digitisation uncertainty is ~ the pixel size in data units
    (record it in ``digitization_sd_mol``)."""
    from PIL import Image

    im = np.asarray(Image.open(image_path).convert("RGB")).astype(float)
    dist = np.sqrt(np.sum((im - np.array(color, float)) ** 2, axis=2))
    mask = dist <= tol
    pts = []
    for x in range(mask.shape[1]):
        ys = np.nonzero(mask[:, x])[0]
        if len(ys) >= min_pixels:
            pts.append((float(x), float(ys.mean())))
    return np.array(pts)


# ---------------------------------------------------------------- leaderboard
def _metrics(y: np.ndarray, pred: np.ndarray, sd: np.ndarray) -> dict[str, float]:
    err = pred - y
    return {"rmse": float(np.sqrt(np.mean(err**2))), "max_err": float(np.max(np.abs(err))), "bias": float(np.mean(err)),
            "coverage95": float(np.mean(np.abs(err) <= 1.96 * sd))}  # magic: 95 % normal quantile


def run_model(level: str, exp: Experiment, params: ParamSet) -> np.ndarray:
    """Predicted cumulative moles at the experiment's times for a model level."""
    ch = exp.channels[0]
    sc = exp.scenario
    if level == "reduced":
        s = lf.simulate_fast(lf.build_constants(sc, params, "empirical"), sc, 2.0)
        return np.interp(ch.t, s["t"], s["gen"])
    if level == "echem_reduced":
        s = lf.simulate_fast(lf.build_constants(sc, params, "echem"), sc, 2.0)
        return np.interp(ch.t, s["t"], s["gen"])
    kw: dict[str, Any] = {}
    if level == "L2":
        kw["fidelity"] = "L2"
    if level == "electrochemical":
        kw["rate_model"] = "electrochemical"
    r = simulate(sc.model_copy(update=kw), params, SolverSettings(dt_out=5.0, rtol=1e-6))
    return np.interp(ch.t, r.t, r["h2_gen_mol"])


LEVELS = ("reduced", "L1", "L2", "electrochemical", "echem_reduced")


def leaderboard(datasets: list[dict[str, Any]], params: ParamSet | None = None, levels: tuple[str, ...] = LEVELS
                ) -> dict[str, Any]:
    params = params or ParamSet()
    board: dict[str, Any] = {"code_version": __version__, "git": git_hash(), "levels": {}, "datasets": [d["id"] for d in datasets]}
    for lv in levels:
        rows = []
        for d in datasets:
            e = to_experiment(d)
            pr = run_model(lv, e, params)
            m = _metrics(e.channels[0].y, pr, e.channels[0].sigma)
            rows.append({"id": d["id"], "T0_C": d["conditions"].get("T0_C"), "c_naoh_M": d["conditions"].get("c_naoh_M"), **m})
        agg = {k: float(np.mean([r[k] for r in rows])) for k in ("rmse", "max_err", "bias", "coverage95")} if rows else {}
        board["levels"][lv] = {"rows": rows, "aggregate": agg}
    return board


def regression_gate(board: dict[str, Any], baseline_path: Path, rel_tol: float = 0.02) -> list[str]:
    """Compare aggregate RMSE per level with the baseline; returns the list of regressions (empty = OK)."""
    if not baseline_path.exists():
        return []
    base = json.loads(baseline_path.read_text())
    bad = []
    for lv, v in board["levels"].items():
        b = base.get("levels", {}).get(lv, {}).get("aggregate", {})
        a = v.get("aggregate", {})
        if b and a and a["rmse"] > b["rmse"] * (1.0 + rel_tol) + 1e-12:  # magic: absolute floor
            bad.append(f"{lv}: RMSE {a['rmse']:.4g} > baseline {b['rmse']:.4g}")
    return bad


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="hydra-benchmarks")
    ap.add_argument("--update-baseline", action="store_true")
    args = ap.parse_args(argv)
    data = load_all()
    board = leaderboard(data)
    print(json.dumps({lv: v["aggregate"] for lv, v in board["levels"].items()}, indent=1))
    path = REPO_BENCH / "baseline.json"
    if args.update_baseline:
        path.write_text(json.dumps(board, indent=1))
    else:
        for line in regression_gate(board, path):
            print("REGRESSION:", line)
