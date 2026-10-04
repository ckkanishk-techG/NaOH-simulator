"""Drive cycles (stack-current or power profiles) for controller comparison, incl. CSV upload."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from ..constants import F


def constant(i_a: float, duration: float) -> list[tuple[float, float]]:
    return [(0.0, i_a)]


def step(i_lo: float, i_hi: float, t_step: float, duration: float) -> list[tuple[float, float]]:
    return [(0.0, i_lo), (t_step, i_hi)]


def ramp(i_lo: float, i_hi: float, t0: float, t1: float, n: int = 12) -> list[tuple[float, float]]:
    ts = np.linspace(t0, t1, n)
    return [(0.0, i_lo)] + [(float(t), float(i_lo + (i_hi - i_lo) * (k + 1) / n)) for k, t in enumerate(ts)]


def stop_go(duration: float, i_max: float = 3.0, seed: int = 0, seg_s: float = 60.0) -> list[tuple[float, float]]:
    """Synthetic urban-style stop-and-go cycle: random plateaus separated by idle periods (seeded)."""
    rng = np.random.default_rng(seed)
    out, t = [(0.0, 0.0)], seg_s
    while t < duration:
        out.append((t, float(rng.choice([0.0, 0.3, 0.6, 1.0, 0.8]) * i_max)))
        t += seg_s * float(rng.uniform(0.5, 1.5))  # magic: segment-length jitter
    return out


def load_csv(path: str | Path, kind: str = "current") -> list[tuple[float, float]]:
    """Uploaded drive cycle: CSV with columns ``t_s`` and ``i_A`` (kind='current') or ``P_W`` (kind='power')."""
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append((float(r["t_s"]), float(r["i_A" if kind == "current" else "P_W"])))
    if not rows:
        raise ValueError("empty drive cycle")
    ts = [a for a, _ in rows]
    if ts != sorted(ts):
        raise ValueError("drive-cycle times must be increasing")
    return rows


def power_to_current(profile_w: list[tuple[float, float]], n_cells: int, v_cell: float = 0.7) -> list[tuple[float, float]]:
    """Convert a power profile to stack current at a nominal cell voltage."""
    return [(t, p / (n_cells * v_cell)) for t, p in profile_w]


def h2_demand_mol_s(current_a: float, n_cells: int) -> float:
    return n_cells * current_a / (2.0 * F)
