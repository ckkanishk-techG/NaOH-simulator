"""Global (Sobol) sensitivity analysis and tornado charts on the reduced model."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core import l1_fast as lf
from ..core.params import ParamSet, bounds
from ..core.scenario import Scenario
from ..thermo import propdb as DB

QOIS = ("h2_mol_at_tref", "t90_s", "peak_T_K", "h2_final_mol")


def _qoi(sim: dict[str, np.ndarray], sc: Scenario, t_ref: float) -> dict[str, float]:
    gen = sim["gen"]
    tot = gen[-1]
    i = int(np.searchsorted(gen, 0.9 * tot)) if tot > 0 else len(gen) - 1  # magic: 90 %
    return {"h2_mol_at_tref": float(np.interp(t_ref, sim["t"], gen)), "t90_s": float(sim["t"][min(i, len(gen) - 1)]),
            "peak_T_K": float(sim["T"].max()), "h2_final_mol": float(tot)}


def _eval(sc: Scenario, base: ParamSet, names: list[str], u: np.ndarray, is_log: np.ndarray, model: str, dt: float,
          t_ref: float) -> dict[str, float]:
    vals = np.where(is_log, np.exp(u), u)
    p = base.with_(**dict(zip(names, vals, strict=True)))
    return _qoi(lf.simulate_fast(lf.build_constants(sc, p, model, dt), sc, dt), sc, t_ref)


def sobol_indices(sc: Scenario, names: list[str], base: ParamSet | None = None, model: str = "empirical",
                  n: int = 128, qoi: str = "h2_mol_at_tref", dt: float = 4.0, t_ref: float | None = None,
                  seed: int = 0) -> dict[str, Any]:
    """First-order and total Sobol indices (Saltelli sampling, SALib) over the DB plausible ranges
    (log-uniform for log-normal-prior parameters)."""
    from SALib.analyze import sobol as asobol
    from SALib.sample import sobol as ssobol

    base = base or ParamSet()
    t_ref = t_ref or 0.5 * sc.duration_s
    lo, hi = bounds(names)
    is_log = np.array([DB.entry(k).dist == "logn" for k in names])
    ulo, uhi = np.where(is_log, np.log(np.maximum(lo, 1e-300)), lo), np.where(is_log, np.log(hi), hi)  # magic: floor
    problem = {"num_vars": len(names), "names": names, "bounds": np.column_stack([ulo, uhi]).tolist()}
    x = ssobol.sample(problem, n, calc_second_order=False, seed=seed)
    y = np.array([_eval(sc, base, names, xi, is_log, model, dt, t_ref)[qoi] for xi in x])
    res = asobol.analyze(problem, y, calc_second_order=False, print_to_console=False, seed=seed)
    return {"names": names, "qoi": qoi, "S1": dict(zip(names, res["S1"], strict=True)),
            "ST": dict(zip(names, res["ST"], strict=True)), "S1_conf": dict(zip(names, res["S1_conf"], strict=True)),
            "ST_conf": dict(zip(names, res["ST_conf"], strict=True)), "n_eval": len(y), "var": float(np.var(y))}


def tornado(sc: Scenario, names: list[str], base: ParamSet | None = None, model: str = "empirical",
            qoi: str = "h2_mol_at_tref", dt: float = 4.0, t_ref: float | None = None, n_sd: float = 1.0
            ) -> list[dict[str, float]]:
    """One-at-a-time response to +/- n_sd prior standard deviations (log scale for log-normal priors),
    sorted by swing."""
    base = base or ParamSet()
    t_ref = t_ref or 0.5 * sc.duration_s
    rows = []
    q0 = _qoi(lf.simulate_fast(lf.build_constants(sc, base, model, dt), sc, dt), sc, t_ref)[qoi]
    for n in names:
        e = DB.entry(n)
        v0 = base[n]
        sd = e.sd or 0.1 * abs(v0)  # magic: default 10 % when no prior sd is known
        lo_v = v0 * np.exp(-n_sd * sd) if e.dist == "logn" else v0 - n_sd * sd
        hi_v = v0 * np.exp(n_sd * sd) if e.dist == "logn" else v0 + n_sd * sd
        out = []
        for v in (lo_v, hi_v):
            q = _qoi(lf.simulate_fast(lf.build_constants(sc, base.with_(**{n: float(v)}), model, dt), sc, dt), sc, t_ref)[qoi]
            out.append(q)
        rows.append({"name": n, "low_value": float(lo_v), "high_value": float(hi_v), "q_low": out[0], "q_high": out[1],
                     "q_base": q0, "swing": abs(out[1] - out[0])})
    return sorted(rows, key=lambda r: -r["swing"])
