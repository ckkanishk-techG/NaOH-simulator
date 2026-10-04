"""Training-data generation for the ML surrogate: Latin-hypercube scenarios simulated in parallel with the
high-fidelity models (full adaptive L1, or the validated reduced model for fast bulk generation)."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.stats import qmc

from ..core import l1_fast as lf
from ..core.l1 import SolverSettings, simulate
from ..core.params import ParamSet
from ..core.scenario import Scenario

FEATURES = ("ln_al_mass", "ln_c_naoh", "ln_v_liq", "T0_C", "T_amb_C", "ln_dim", "is_foil")


@dataclass
class InputSpace:
    """Validity box of the surrogate (also its domain-of-validity box)."""

    al_mass_g: tuple[float, float] = (0.3, 8.0)
    c_naoh_M: tuple[float, float] = (0.5, 5.0)
    v_liq_mL: tuple[float, float] = (80.0, 300.0)
    T0_C: tuple[float, float] = (15.0, 40.0)
    T_amb_C: tuple[float, float] = (15.0, 35.0)
    dim_um: tuple[float, float] = (10.0, 300.0)
    forms: tuple[str, ...] = ("foil", "powder")
    mode: str = "open"
    duration_s: float = 3600.0
    n_time: int = 61
    peak_T_max_C: float = 90.0  # beyond this the reduced physics (no boiling) and the surrogate are not valid

    def lo_hi(self) -> tuple[np.ndarray, np.ndarray]:
        lo = np.array([np.log(self.al_mass_g[0]), np.log(self.c_naoh_M[0]), np.log(self.v_liq_mL[0]), self.T0_C[0],
                       self.T_amb_C[0], np.log(self.dim_um[0]), 0.0])
        hi = np.array([np.log(self.al_mass_g[1]), np.log(self.c_naoh_M[1]), np.log(self.v_liq_mL[1]), self.T0_C[1],
                       self.T_amb_C[1], np.log(self.dim_um[1]), 1.0])
        return lo, hi

    @property
    def t_grid(self) -> np.ndarray:
        return np.linspace(0.0, self.duration_s, self.n_time)


def features_of(sc: Scenario) -> np.ndarray:
    return np.array([np.log(sc.al_mass_g), np.log(sc.c_naoh_M), np.log(sc.v_liq_mL), sc.T0_C, sc.T_amb_C, np.log(sc.dim_um),
                     1.0 if sc.form == "foil" else 0.0])


def scenario_of(x: np.ndarray, space: InputSpace) -> Scenario:
    foil = x[6] > 0.5
    return Scenario(al_mass_g=float(np.exp(x[0])), c_naoh_M=float(np.exp(x[1])), v_liq_mL=float(np.exp(x[2])), T0_C=float(x[3]),
                    T_amb_C=float(x[4]), dim_um=float(np.exp(x[5])), form="foil" if foil else "powder",  # type: ignore[arg-type]
                    v_vessel_mL=max(2.5 * float(np.exp(x[2])), float(np.exp(x[2])) + 150.0), mode=space.mode,  # type: ignore[arg-type]
                    duration_s=space.duration_s, vent_diameter_mm=10.0)


def sample_inputs(n: int, space: InputSpace, seed: int = 0) -> np.ndarray:
    lo, hi = space.lo_hi()
    u = qmc.LatinHypercube(d=len(lo), seed=seed).random(n)
    x = lo + u * (hi - lo)
    x[:, 6] = (u[:, 6] > 0.5).astype(float)  # form is categorical
    return x


def extended_features(sc: Scenario, params: ParamSet | None = None) -> np.ndarray:
    """Design features plus physics-derived scales: ln tau0 (isothermal dissolution time at the initial state),
    adiabatic temperature rise and the Semenov-type number dT_ad*Ea/(R T0^2). Depends on the kinetic parameters,
    so a surrogate is only valid for the parameter set it was trained with."""
    from ..constants import R as _R

    p = params or ParamSet()
    k = lf.build_constants(sc, p, dt=4.0)  # magic: step only affects the (unused) ramp width
    t0 = sc.T0_C + 273.15
    kr = p["k25"] * np.exp(-p["Ea"] / _R * (1.0 / t0 - 1.0 / 298.15)) * sc.c_naoh_M ** p["n_oh"]  # magic: T_ref
    tau0 = k[lf.KI["n0"]] / (k[lf.KI["a0"]] * kr)
    dtad = -k[lf.KI["dh298"]] * k[lf.KI["n0"]] / (k[lf.KI["nW0"]] * k[lf.KI["cp_w"]] + k[lf.KI["C_wall"]])
    return np.concatenate([features_of(sc), [np.log(tau0), dtad, dtad * p["Ea"] / (_R * t0**2)]])


def run_one(args: tuple[np.ndarray, InputSpace, str, dict[str, float]]) -> dict[str, np.ndarray]:
    """Simulate one design; returns the curves on the surrogate time grid (module-level for multiprocessing)."""
    x, space, fidelity, overrides = args
    sc = scenario_of(x, space)
    p = ParamSet(overrides)
    t = space.t_grid
    nmax = 1.5 * sc.al_mass_g * 1e-3 / 26.9815385e-3  # magic: g/mol Al
    if fidelity == "fast":
        s = lf.simulate_fast(lf.build_constants(sc, p, dt=4.0), sc, dt=4.0)  # magic: step
        n = np.interp(t, s["t"], s["gen"])
        temp = np.interp(t, s["t"], s["T"])
        pg = np.zeros_like(t)
    else:
        r = simulate(sc, p, SolverSettings(dt_out=float(t[1] - t[0]), rtol=1e-6))
        n = np.interp(t, r.t, r["h2_gen_mol"])
        temp = np.interp(t, r.t, r["T"])
        pg = np.interp(t, r.t, (r["P"] - 101325.0) / 1e5)
    xx = np.clip(n / nmax, 0.0, 1.0)
    th = float(np.interp(0.5, np.maximum.accumulate(xx), t)) if xx[-1] >= 0.5 else float("nan")  # magic: half conversion
    return {"X": xx, "dT": temp - (sc.T0_C + 273.15), "Pg": pg, "nmax": np.array(nmax), "t_half": np.array(th),
            "feat": extended_features(sc, p), "peak_T_C": np.array(float(temp.max() - 273.15))}


def generate_dataset(n: int, space: InputSpace | None = None, fidelity: str = "fast", workers: int | None = None, seed: int = 0,
                     overrides: dict[str, float] | None = None) -> dict[str, Any]:
    """Run ``n`` simulations in parallel (``workers`` processes; default = CPU count - 1, capped by n)."""
    space = space or InputSpace()
    x = sample_inputs(n, space, seed)
    jobs = [(xi, space, fidelity, overrides or {}) for xi in x]
    workers = workers or max(1, min((os.cpu_count() or 2) - 1, n))
    if workers == 1:
        out = [run_one(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            out = list(ex.map(run_one, jobs, chunksize=max(1, n // (4 * workers))))  # magic: chunking
    d = {"x": x, "X": np.array([o["X"] for o in out]), "dT": np.array([o["dT"] for o in out]),
         "Pg": np.array([o["Pg"] for o in out]), "nmax": np.array([float(o["nmax"]) for o in out]), "space": space,
         "fidelity": fidelity, "t_half": np.array([float(o["t_half"]) for o in out]),
         "feat": np.array([o["feat"] for o in out]), "peak_T_C": np.array([float(o["peak_T_C"]) for o in out]),
         "params": overrides or {}}
    valid = np.isfinite(d["t_half"]) & (d["peak_T_C"] <= space.peak_T_max_C)
    return {k: (v[valid] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in d.items()} | {"n_requested": n}
