"""Dosing plant for control studies: Al charge in a small, concentrated-NaOH-fed vessel feeding a stack load.

Manipulated variable: NaOH-solution dosing flow ``u`` [mL/min] at concentration ``c_dose``. Measurements: vessel
pressure, liquid/wall temperature, liquid volume, hydrogen-delivery fraction. The plant is the full adaptive L1
engine (valve events, boiling clamp, energy balance, stack gating) stepped with a zero-order hold.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..constants import F, MW_H2O, P_ATM
from ..core.l1 import L1Model, SolverSettings
from ..core.params import ParamSet
from ..core.scenario import Scenario


@dataclass
class Obs:
    t: float
    P_bar_g: float
    T_C: float
    Tw_C: float
    V_liq_mL: float
    c_oh: float
    delivered: float  # fraction of the requested H2 delivered in the last step (1 = no starvation)
    demand_mol_s: float
    gen_mol_s: float
    sf: float
    boiling: bool


def drive_scenario(load_steps: list[tuple[float, float]], duration_s: float, al_mass_g: float = 10.0, dim_um: float = 40.0,
                   v_liq0_mL: float = 40.0, c0_M: float = 0.3, cells: int = 4, p_relief_bar_g: float = 1.5, cooling_UA: float = 2.0,
                   **kw: Any) -> Scenario:
    return Scenario(al_mass_g=al_mass_g, form="powder", dim_um=dim_um, c_naoh_M=c0_M, v_liq_mL=v_liq0_mL, v_vessel_mL=500.0,
                    mode="regulated", p_relief_bar_g=p_relief_bar_g, cells=cells, load_steps=load_steps, duration_s=duration_s,
                    cooling_UA_W_K=cooling_UA, **kw)


class DosingPlant:
    def __init__(self, sc: Scenario, params: ParamSet | None = None, c_dose_M: float = 6.0, dt: float = 2.0,
                 u_max_mL_min: float = 5.0, headspace: str = "h2") -> None:
        self.sc, self.p = sc.model_copy(deep=True), params or ParamSet()
        self.headspace = headspace
        self.c_dose, self.dt, self.u_max = c_dose_M, dt, u_max_mL_min
        self.m = L1Model(self.sc, self.p, SolverSettings(rtol=1e-6, atol=1e-9, max_step=dt))
        self.reset()

    def reset(self) -> None:
        self.m.events = {}
        self.m.valve_open = False
        self.m.n_valve_events = 0
        self.y = self.m.initial_state()
        if self.headspace == "h2":  # headspace already flushed with hydrogen (after a first start / inert purge)
            ix = self.m.idx
            self.y[ix["nH2"]], self.y[ix["nAir"]] = self.y[ix["nAir"]], 0.0
        self.t = 0.0
        self.total_dosed_mL = 0.0
        self.log: list[dict[str, float]] = []

    def snapshot(self) -> tuple[Any, ...]:
        return (self.y.copy(), self.t, self.m.valve_open, self.total_dosed_mL, copy.copy(self.log))

    def restore(self, snap: tuple[Any, ...]) -> None:
        self.y, self.t, self.m.valve_open, self.total_dosed_mL, self.log = snap[0].copy(), snap[1], snap[2], snap[3], copy.copy(snap[4])

    def _u_to_engine(self, u_mL_min: float) -> tuple[float, float]:
        rho = self.m.rho_w
        mol_w_s = u_mL_min * 1e-6 / 60.0 * rho / MW_H2O  # water moles/s carried by the dose (solution ~ water-like density)
        return mol_w_s, self.c_dose

    def liquid_volume_mL(self) -> float:
        return self.m.liquid(self.y)[0] * 1e6

    def step(self, u_mL_min: float) -> Obs:
        u = float(np.clip(u_mL_min, 0.0, self.u_max))
        self.m.u = self._u_to_engine(u)
        cx = self.m.cidx
        y0 = self.y
        _, ys = self.m.integrate(y0, self.t, self.t + self.dt)
        y1 = ys[:, -1].copy()
        self.y, self.t = y1, self.t + self.dt
        self.total_dosed_mL += u * self.dt / 60.0
        d_dem, d_del = y1[cx["I_demand_t"]] - y0[cx["I_demand_t"]], y1[cx["I_deliv_t"]] - y0[cx["I_deliv_t"]]
        deliv = d_del / d_dem if d_dem > 1e-12 else 1.0  # magic: guard
        demand = self.sc.cells * (d_dem / self.dt) / (2.0 * F)
        gen = (y1[cx["gen_H2"]] - y0[cx["gen_H2"]]) / self.dt
        v_l, c_oh, _ = self.m.liquid(y1)
        p = self.m.pressure(y1)
        pg = max(p - P_ATM, 0.0)
        sf = 99.0 if pg < 1.0 else min(self.p["sy23"] / (pg * self.p["r_ves"] / self.p["t_ves"]), 99.0)  # magic: cap
        obs = Obs(self.t, (p - P_ATM) / 1e5, y1[self.m.idx["T"]] - 273.15, y1[self.m.idx["Tw"]] - 273.15, v_l * 1e6, c_oh,
                  float(min(max(deliv, 0.0), 1.0)), demand, gen, sf, bool(y1[cx["boil_mol"]] > y0[cx["boil_mol"]]))
        self.log.append({"t": obs.t, "u": u, "P": obs.P_bar_g, "T": obs.T_C, "Tw": obs.Tw_C, "V": obs.V_liq_mL, "c": obs.c_oh,
                         "deliv": obs.delivered, "demand": obs.demand_mol_s, "gen": obs.gen_mol_s, "sf": obs.sf})
        return obs

    def measure(self) -> Obs:
        """Observation at the current state without advancing time."""
        y = self.y
        v_l, c_oh, _ = self.m.liquid(y)
        p = self.m.pressure(y)
        pg = max(p - P_ATM, 0.0)
        sf = 99.0 if pg < 1.0 else min(self.p["sy23"] / (pg * self.p["r_ves"] / self.p["t_ves"]), 99.0)  # magic: cap
        return Obs(self.t, (p - P_ATM) / 1e5, y[self.m.idx["T"]] - 273.15, y[self.m.idx["Tw"]] - 273.15, v_l * 1e6, c_oh, 1.0, 0.0, 0.0, sf,
                   False)
