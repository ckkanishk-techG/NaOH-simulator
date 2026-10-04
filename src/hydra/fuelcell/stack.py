"""Modular AEM stack: series/parallel cells, cell spread, thermal coupling, water and ageing states."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..constants import F
from ..core.params import ParamSet
from . import flowfield
from .cell import AEMCell, CellCond


@dataclass
class StackSpec:
    n_series: int = 4
    n_parallel: int = 1
    cell_area_cm2: float = 25.0
    seed: int = 0


class Stack:
    """Series stack (``n_parallel`` identical strings share the total current)."""

    def __init__(self, spec: StackSpec, params: ParamSet | None = None) -> None:
        self.spec = spec
        self.p = params or ParamSet()
        self.cell = AEMCell(self.p, spec.cell_area_cm2)
        rng = np.random.default_rng(spec.seed)
        n = spec.n_series
        spread = self.p["fc_asr_spread"]
        asr_mult = np.clip(1.0 + spread * rng.standard_normal(n), 0.5, 2.0) if spread > 0 else np.ones(n)  # magic: clip
        flow = flowfield.sheet_flow_factors(self.p, n) if n > 1 else np.ones(1)
        self.flow_factors = flow
        self.cells = [CellCond(T=self.p["fc_T_stack"], lam=self.p["fc_lambda_ref"], asr_mult=float(asr_mult[k]),
                               iL_scale=float(min(flow[k], 1.0))) for k in range(n)]
        self.T_amb = 298.15  # magic: overwritten by the system model
        self.t_h = 0.0  # operating hours (ageing)

    @property
    def n_cells(self) -> int:
        return self.spec.n_series * self.spec.n_parallel

    def i_density(self, i_total: float) -> float:
        """Current density [A m-2] for stack terminal current ``i_total`` [A]."""
        return i_total / self.spec.n_parallel / self.cell.area

    def cell_voltages(self, i_total: float, p_h2_atm: float = 1.0) -> np.ndarray:
        i = self.i_density(i_total)
        return np.array([self.cell.voltage(i, c, p_h2_atm)["V"] for c in self.cells])

    def voltage(self, i_total: float, p_h2_atm: float = 1.0) -> float:
        return float(self.cell_voltages(i_total, p_h2_atm).sum())

    def power(self, i_total: float, p_h2_atm: float = 1.0) -> float:
        return self.voltage(i_total, p_h2_atm) * i_total

    def max_power_point(self, p_h2_atm: float = 1.0) -> tuple[float, float]:
        """(I_mpp [A], P_max [W]) by golden-section search over the stack current."""
        i_max = self.cell.limiting_current(self.cells[0]) * self.cell.area * self.spec.n_parallel * 0.999  # magic: below i_L
        lo, hi = 0.0, i_max
        g = 0.6180339887  # magic: golden ratio conjugate
        for _ in range(60):  # magic: iterations
            a, b = hi - g * (hi - lo), lo + g * (hi - lo)
            if self.power(a, p_h2_atm) < self.power(b, p_h2_atm):
                lo = a
            else:
                hi = b
        i = 0.5 * (lo + hi)
        return i, self.power(i, p_h2_atm)

    def current_for_power(self, p_target: float, p_h2_atm: float = 1.0) -> tuple[float, bool]:
        """Stack current delivering ``p_target`` [W] (lower-current branch). Returns (I, feasible)."""
        if p_target <= 0.0:
            return 0.0, True
        i_mpp, p_max = self.max_power_point(p_h2_atm)
        if p_target >= p_max:
            return i_mpp, False
        lo, hi = 0.0, i_mpp
        for _ in range(60):  # magic: bisection iterations
            mid = 0.5 * (lo + hi)
            if self.power(mid, p_h2_atm) < p_target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi), True

    def h2_flow(self, i_total: float) -> float:
        """H2 consumption [mol/s] at stack current ``i_total`` (all cells)."""
        return self.n_cells * i_total / self.spec.n_parallel / (2.0 * F) / self.p["fc_utilization"]

    # --- state evolution ------------------------------------------------------------------------
    def step(self, dt: float, i_total: float, rh_anode: float, p_h2_atm: float = 1.0, h2_flow_m2: float | None = None) -> None:
        """Advance water content, carbonation, ageing and (optionally) temperatures by ``dt`` seconds."""
        p = self.p
        i = self.i_density(i_total)
        volt = [self.cell.voltage(i, c, p_h2_atm)["V"] for c in self.cells]
        e_th = self.p["hhv_h2"] / (2.0 * F)
        n = len(self.cells)
        if p["fc_thermal_on"] > 0.5:  # magic: switch
            t_arr = np.array([c.T for c in self.cells])
            q_gen = np.array([(e_th - volt[k]) * i_total / self.spec.n_parallel for k in range(n)])
            sub = max(int(np.ceil(dt / 2.0)), 1)  # magic: explicit sub-steps of <= 2 s
            h = dt / sub
            for _ in range(sub):
                lap = np.zeros(n)
                for k in range(n):
                    if k > 0:
                        lap[k] += t_arr[k - 1] - t_arr[k]
                    if k < n - 1:
                        lap[k] += t_arr[k + 1] - t_arr[k]
                ua = p["fc_UA_cell"] * np.ones(n)
                ua[0] *= p["fc_UA_end_mult"]
                ua[-1] *= p["fc_UA_end_mult"]
                t_arr = t_arr + h * (q_gen - ua * (t_arr - self.T_amb) + p["fc_k_cc"] * lap) / p["fc_cell_C"]
            for k, c in enumerate(self.cells):
                c.T = float(t_arr[k])
        for c in self.cells:
            dlam, _ = self.cell.water_rates(i, c, rh_anode, p["fc_RH_air"], h2_flow_m2)
            c.lam = float(min(max(c.lam + dt * dlam, 0.3), p["fc_lambda_max"] * 1.2))  # magic: bounds
            c.x_carb = float(min(max(c.x_carb + dt * self.cell.carb_rate(i, c), 0.0), 0.99))  # magic: cap
        self.t_h += dt / 3600.0
        for c in self.cells:
            c.age_cat = float(np.exp(-p["fc_k_cat"] * self.t_h))
            c.age_mem = float(1.0 + p["fc_k_mem"] * self.t_h)

    def spread(self, i_total: float) -> dict[str, float]:
        v = self.cell_voltages(i_total)
        return {"v_min": float(v.min()), "v_max": float(v.max()), "v_std": float(v.std()),
                "spread_mV": float(1.0e3 * (v.max() - v.min()))}

