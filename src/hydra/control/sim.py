"""Closed-loop simulation, interlocks and controller comparison on drive cycles."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

from ..core.params import ParamSet
from ..core.scenario import Scenario
from ..thermo import propdb as DB
from .plant import DosingPlant, Obs


class Controller(Protocol):
    def update(self, t: float, obs: Obs) -> float: ...
    def reset(self) -> None: ...


@dataclass
class Interlock:
    """Software safety interlock on the dosing command (sits in front of ANY controller output)."""

    T_max_C: float = 75.0
    P_max_bar_g: float = 1.4
    sf_min: float = 3.0
    V_max_mL: float = 350.0
    tripped: list[str] = field(default_factory=list)

    def filter(self, u: float, obs: Obs) -> float:
        why = []
        if obs.T_C > self.T_max_C:
            why.append("T")
        if obs.P_bar_g > self.P_max_bar_g:
            why.append("P")
        if obs.sf < self.sf_min:
            why.append("SF")
        if obs.V_liq_mL > self.V_max_mL:
            why.append("V")
        if obs.boiling:
            why.append("boil")
        if why:
            self.tripped.append(f"t={obs.t:.0f}s:" + ",".join(why))
            return 0.0
        return u


@dataclass
class ClosedLoopResult:
    t: np.ndarray
    series: dict[str, np.ndarray]
    metrics: dict[str, float]
    interlock_trips: list[str]


def run_closed_loop(plant: DosingPlant, controller: Controller | Callable[[float, Obs], float], duration_s: float,
                    setpoint: float, interlock: Interlock | None = None, t_skip_s: float = 0.0) -> ClosedLoopResult:
    plant.reset()
    if hasattr(controller, "reset"):
        controller.reset()
    obs = plant.measure()
    lock = interlock or Interlock()
    rows = []
    for _ in range(int(duration_s / plant.dt)):
        u = controller.update(plant.t, obs) if hasattr(controller, "update") else controller(plant.t, obs)
        u = lock.filter(u, obs)
        obs = plant.step(u)
        rows.append((obs.t, u, obs.P_bar_g, obs.T_C, obs.Tw_C, obs.V_liq_mL, obs.delivered, obs.demand_mol_s, obs.gen_mol_s, obs.sf))
    a = np.array(rows)
    keys = ("t", "u", "P", "T", "Tw", "V", "delivered", "demand", "gen", "sf")
    s = {k: a[:, i] for i, k in enumerate(keys)}
    m = s["t"] >= t_skip_s
    dt = plant.dt
    dem = s["demand"] > 0
    starve = (s["delivered"] < 0.98) & dem  # magic: starvation threshold
    cx = plant.m.cidx
    vent = plant.y[cx["vent_H2"]] + plant.y[cx["purge_H2"]]
    gen = plant.y[cx["gen_H2"]]
    metrics = {
        "rms_pressure_error_bar": float(np.sqrt(np.mean((s["P"][m] - setpoint) ** 2))),
        "iae_bar_s": float(np.sum(np.abs(s["P"][m] - setpoint)) * dt),
        "starved_s": float(np.sum(starve[m]) * dt),
        "peak_T_C": float(s["T"].max()),
        "peak_Tw_C": float(s["Tw"].max()),
        "peak_P_bar_g": float(s["P"].max()),
        "min_sf": float(s["sf"].min()),
        "max_V_mL": float(s["V"].max()),
        "dosed_mL": float(plant.total_dosed_mL),
        "vented_fraction": float(vent / gen) if gen > 0 else 0.0,
        "delivered_fraction_mean": float(np.mean(s["delivered"][m & dem])) if np.any(m & dem) else 1.0,
        "safety_margin_T_C": float(DB.get("T_hdpe_max") - 273.15 - s["Tw"].max()),
        "interlock_trips": float(len(lock.tripped)),
    }
    return ClosedLoopResult(s["t"], s, metrics, lock.tripped)


def compare_controllers(make_plant: Callable[[], DosingPlant], controllers: dict[str, Controller | Callable[[float, Obs], float]],
                        duration_s: float, setpoint: float, t_skip_s: float = 120.0) -> dict[str, ClosedLoopResult]:
    """Run every controller on a fresh plant (same drive cycle) and return the results keyed by name."""
    return {n: run_closed_loop(make_plant(), c, duration_s, setpoint, Interlock(), t_skip_s) for n, c in controllers.items()}


def metrics_table(results: dict[str, ClosedLoopResult]) -> list[dict[str, Any]]:
    return [{"controller": n, **r.metrics} for n, r in results.items()]


_ = (ParamSet, Scenario)
