r"""Full system: generator (L1/L2 vessel) -> conditioning -> AEM stack -> DC/DC -> load (+ battery buffer).

The vessel engine and the stack are coupled in time steps of ``sys_dt``: the load fixes the stack
current command; the engine returns the *delivered* fraction of the requested H2 (starvation); the
stack states (water, carbonation, ageing, temperature) advance with the actual current; the optional
battery covers shortfall or smooths the stack with a first-order feed-forward filter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from pydantic import BaseModel, Field

from ..constants import KELVIN_OFFSET, F
from ..core.l1 import L1Model, SimResult, SolverSettings, interp_step
from ..core.params import ParamSet
from ..core.scenario import Scenario
from ..fuelcell.bop import DCDC, Battery, Conditioner
from ..fuelcell.stack import Stack, StackSpec
from ..thermo import propdb as DB
from ..thermo.ranges import collect


class SystemSpec(BaseModel):
    n_series: int = Field(4, ge=1)
    n_parallel: int = Field(1, ge=1)
    cell_area_cm2: float = Field(25.0, gt=0)
    load_kind: str = "current"  # 'current' (per-cell steps from Scenario.load_steps) or 'power'
    load_power_W: list[tuple[float, float]] = Field(default_factory=list)  # (t, W) if load_kind == 'power'
    battery_Wh: float = Field(0.0, ge=0)
    battery_p_max_W: float = Field(50.0, gt=0)
    battery_soc0: float = Field(0.5, ge=0, le=1)
    feedforward_tau_s: float = Field(60.0, gt=0)
    condenser: bool = True
    dryer: bool = False
    seed: int = 0


@dataclass
class SystemResult:
    t: np.ndarray
    series: dict[str, np.ndarray]
    summary: dict[str, float]
    vessel: SimResult
    warnings: list[str] = field(default_factory=list)
    sankey: dict[str, list] = field(default_factory=dict)

    def __getitem__(self, k: str) -> np.ndarray:
        return self.series[k]


class SystemModel:
    def __init__(self, sc: Scenario, spec: SystemSpec | None = None, params: ParamSet | None = None,
                 settings: SolverSettings | None = None) -> None:
        self.spec = spec or SystemSpec()
        self.p = params or ParamSet()
        n = self.spec.n_series * self.spec.n_parallel
        self.sc = sc.model_copy(update={"mode": "regulated", "cells": n, "cell_area_cm2": self.spec.cell_area_cm2,
                                        "load_steps": list(sc.load_steps)}, deep=True)
        self.src_steps = list(sc.load_steps)
        self.settings = settings or SolverSettings()
        self.vessel = L1Model(self.sc, self.p, self.settings)
        self.stack = Stack(StackSpec(n_series=self.spec.n_series, n_parallel=self.spec.n_parallel,
                                     cell_area_cm2=self.spec.cell_area_cm2, seed=self.spec.seed), self.p)
        self.stack.T_amb = sc.T_amb_C + KELVIN_OFFSET
        self.dcdc = DCDC(self.p)
        self.cond = Conditioner(self.p, self.spec.condenser, self.spec.dryer)
        self.batt = Battery(self.spec.battery_Wh, self.spec.battery_p_max_W, self.spec.battery_soc0)

    # ------------------------------------------------------------------ command
    def _command(self, t: float, p_ff: float) -> tuple[float, float, bool]:
        """(stack current [A], load power target [W] (nan for current load), feasible)."""
        sp = self.spec
        if sp.load_kind == "current":
            return interp_step(self.src_steps, t) * sp.n_parallel, float("nan"), True
        p_load = interp_step(sp.load_power_W, t)
        p_target = p_ff if self.batt.capacity_Wh > 0 else p_load
        i, ok = self.stack.current_for_power(self.dcdc.p_in(p_target))
        return i, p_load, ok

    # ------------------------------------------------------------------ run
    def run(self) -> SystemResult:
        with collect() as viol:
            out = self._run()
        out.vessel.diagnostics["range_violations"] = [str(v) for v in viol.values()]
        return out

    def _run(self) -> SystemResult:
        sc, sp, p = self.sc, self.spec, self.p
        dt = DB.get("sys_dt")
        m = self.vessel
        cx = m.cidx
        m.events = {}
        m.n_valve_events = 0
        m.valve_open = False
        y = m.initial_state()
        n_steps = int(round(sc.duration_s / dt))
        rec: dict[str, list[float]] = {k: [] for k in (
            "t", "I", "V", "P_stack", "P_out", "P_load", "eta_stack", "h2_demand", "h2_flow", "phi", "lam", "T_stack",
            "soc", "P_batt", "starve", "rh_anode", "mist_g_s", "flood", "dry", "spread_mV", "unmet_W", "dp_foam")}
        ys = [y.copy()]
        ts = [0.0]
        p_ff = 0.0
        e_unmet = 0.0
        batt_swing = 0.0
        cum_mismatch = 0.0
        cum_min = cum_max = 0.0
        lhv_cell = DB.get("lhv_h2") / (2.0 * F)
        area_flow = self.stack.cell.area * sp.n_parallel
        for k in range(n_steps):
            t0, t1 = k * dt, (k + 1) * dt
            i_cmd, p_load, feasible = self._command(t0, p_ff)
            i_string = i_cmd / sp.n_parallel
            sc.load_steps = [(t0, i_string / p["fc_utilization"])]
            t_arr, y_arr = m.integrate(y, t0, t1)
            y_new = y_arr[:, -1].copy()
            d_dem = y_new[cx["I_demand_t"]] - y[cx["I_demand_t"]]
            d_del = y_new[cx["I_deliv_t"]] - y[cx["I_deliv_t"]]
            phi = d_del / d_dem if d_dem > 1e-12 else 1.0  # magic: guard
            i_act = i_cmd * min(max(phi, 0.0), 1.0)
            v = self.stack.voltage(i_act)
            p_st = v * i_act
            p_dc_out = self.dcdc.p_out(p_st)
            p_b = 0.0
            if sp.load_kind == "power":
                p_short = p_load - p_dc_out
                p_b = self.batt.exchange(p_short, dt, p)
                unmet = max(p_load - p_dc_out - max(p_b, 0.0), 0.0)
                # surplus (stack > load with battery buffer) charges the battery
                if self.batt.capacity_Wh > 0 and p_short < 0:
                    p_b = self.batt.exchange(p_short, dt, p)
                p_ff += (p_load - p_ff) * dt / sp.feedforward_tau_s
            else:
                unmet = 0.0
            e_unmet += unmet * dt
            mism = (p_load if sp.load_kind == "power" else p_dc_out) - p_dc_out
            cum_mismatch += mism * dt
            cum_min, cum_max = min(cum_min, cum_mismatch), max(cum_max, cum_mismatch)
            # conditioning and stack state update
            nH2, nAir, nV = (max(y_new[m.idx[s]], 0.0) for s in ("nH2", "nAir", "nV"))
            tot = max(nH2 + nAir + nV, 1e-30)  # magic: guard
            boil_rate = (y_new[cx["boil_mol"]] - y[cx["boil_mol"]]) / dt
            h2_flow = self.stack.h2_flow(i_act)
            cg = self.cond.condition(y_new[m.idx["T"]], nV / tot, m.pressure(y_new), h2_flow / max(nH2 / tot, 1e-3),  # magic: floor
                                     boil_rate, float(np.mean([c.T for c in self.stack.cells])))
            self.stack.step(dt, i_act, cg.rh_stack, h2_flow_m2=h2_flow / (self.stack.n_cells * area_flow / sp.n_parallel + 1e-12))  # magic: floor
            lam = np.array([c.lam for c in self.stack.cells])
            fl = max(self.stack.cell.flood_risk(float(x)) for x in lam)
            dr = max(self.stack.cell.dry_risk(float(x)) for x in lam)
            cell_v = v / sp.n_series
            vals = {"t": t1, "I": i_act, "V": v, "P_stack": p_st, "P_out": p_dc_out,
                    "P_load": p_load if sp.load_kind == "power" else p_dc_out,
                    "eta_stack": cell_v / lhv_cell * p["fc_utilization"] if i_act > 0 else 0.0,
                    "h2_demand": self.stack.h2_flow(i_cmd), "h2_flow": h2_flow, "phi": phi, "lam": float(lam.mean()),
                    "T_stack": float(np.mean([c.T for c in self.stack.cells])), "soc": self.batt.soc,
                    "P_batt": p_b, "starve": float(phi < 0.98 and i_cmd > 0 or not feasible),  # magic: starvation threshold
                    "rh_anode": cg.rh_stack, "mist_g_s": cg.mist_g_s, "flood": fl, "dry": dr,
                    "spread_mV": self.stack.spread(i_act)["spread_mV"] if i_act > 0 else 0.0, "unmet_W": unmet,
                    "dp_foam": 0.0}
            for kk, vv in vals.items():
                rec[kk].append(vv)
            ys.append(y_new.copy())
            ts.append(t1)
            y = y_new
            batt_swing = cum_max - cum_min
        t_arr = np.array(ts)
        ys_arr = np.array(ys).T
        series_v = m.observables(t_arr, ys_arr)
        res_v = SimResult(t=t_arr, series=series_v, summary={}, ledger=m.ledger(ys_arr), diagnostics={},
                          scenario=sc, y=ys_arr, model=m, events=dict(m.events))
        res_v.summary = m.summarize(res_v)
        h2_stack = float(ys_arr[cx["stack_H2"], -1])
        res_v.diagnostics["n_valve_events"] = m.n_valve_events
        series = {k: np.array(v) for k, v in rec.items()}
        t = series["t"]
        wh = lambda x: float(np.sum(x) * dt / 3600.0)  # noqa: E731
        al_g = sc.al_mass_g * res_v.summary["conversion_pct"] / 100.0
        summary = {
            "Wh_load": wh(series["P_out"]), "Wh_stack": wh(series["P_stack"]), "Wh_unmet": e_unmet / 3600.0,
            "Wh_per_g_Al_reacted": wh(series["P_out"]) / al_g if al_g > 0 else 0.0,
            "Wh_per_g_Al_loaded": wh(series["P_out"]) / sc.al_mass_g,
            "stack_efficiency_energy": wh(series["P_stack"]) * 3600.0 / max(h2_stack * DB.get("lhv_h2"), 1e-12),  # magic: floor
            "system_efficiency": wh(series["P_out"]) * 3600.0 / max(res_v.series["h2_gen_mol"][-1] * DB.get("lhv_h2"), 1e-9),  # magic: floor
            "starve_s": float(np.sum(series["starve"]) * dt),
            "buffer_Wh_needed": batt_swing / 3600.0,
            "mist_g": float(np.sum(series["mist_g_s"]) * dt),
            "flood_max": float(series["flood"].max()), "dry_max": float(series["dry"].max()),
            "peak_P_stack_W": float(series["P_stack"].max()),
            "h2_vented_fraction": res_v.summary["vent_pct"] / 100.0,
        }
        warns: list[str] = []
        if summary["starve_s"] > 0:
            warns.append(f"Stack starved of H2 for {summary['starve_s']:.0f} s (demand exceeded delivery).")
        if sp.load_kind == "power" and e_unmet > 0:
            warns.append(f"Load not met for {e_unmet / 3600.0:.2f} Wh.")
        if sp.load_kind == "power" and self.batt.capacity_Wh < summary["buffer_Wh_needed"] and summary["buffer_Wh_needed"] > 0:
            warns.append(f"Battery buffer of {self.batt.capacity_Wh:.1f} Wh is below the {summary['buffer_Wh_needed']:.1f} Wh "
                         "swing needed to ride through the generator's mismatch.")
        if summary["flood_max"] > 0.5:  # magic: warning thresholds
            warns.append("Membrane/anode flooding risk (water content above flooding onset).")
        if summary["dry_max"] > 0.5:  # magic: warning thresholds
            warns.append("Membrane drying risk (water content below dry-out threshold).")
        out = SystemResult(t=t, series=series, summary=summary, vessel=res_v, warnings=warns)
        out.sankey = sankey(res_v, series, dt)
        return out


def sankey(vessel: SimResult, series: dict[str, np.ndarray], dt: float) -> dict[str, list]:
    """Energy-flow Sankey data (J): reaction heat, H2 chemical energy, stack, DC/DC, load."""
    m = vessel.model
    cx = m.cidx
    assert vessel.y is not None
    y0, y1 = vessel.y[:, 0], vessel.y[:, -1]
    lhv = DB.get("lhv_h2")
    d = lambda k: float(y1[cx[k]] - y0[cx[k]])  # noqa: E731
    q_rx = d("Q_rxn")
    e_loss = d("E_amb")
    e_h2 = d("gen_H2") * lhv
    e_vent = (d("vent_H2") + d("purge_H2")) * lhv
    e_stack_h2 = d("stack_H2") * lhv
    e_el = float(np.sum(series["P_stack"]) * dt)
    e_out = float(np.sum(series["P_out"]) * dt)
    nodes = ["Al hydrolysis heat", "H2 chemical energy (LHV)", "Vessel heat loss", "Heat stored / carried by gas",
             "H2 vented/purged", "H2 retained in vessel", "H2 to stack", "Stack heat", "Stack electricity",
             "DC/DC loss", "Electrical output"]
    ix = {n: i for i, n in enumerate(nodes)}
    link = lambda a, b, v: {"source": ix[a], "target": ix[b], "value": max(v, 0.0)}  # noqa: E731
    links = [
        link("Al hydrolysis heat", "Vessel heat loss", e_loss),
        link("Al hydrolysis heat", "Heat stored / carried by gas", q_rx - e_loss),
        link("H2 chemical energy (LHV)", "H2 vented/purged", e_vent),
        link("H2 chemical energy (LHV)", "H2 to stack", e_stack_h2),
        link("H2 chemical energy (LHV)", "H2 retained in vessel", e_h2 - e_vent - e_stack_h2),
        link("H2 to stack", "Stack electricity", e_el),
        link("H2 to stack", "Stack heat", e_stack_h2 - e_el),
        link("Stack electricity", "Electrical output", e_out),
        link("Stack electricity", "DC/DC loss", e_el - e_out),
    ]
    return {"nodes": nodes, "links": links, "unit": ["J"], "values": [q_rx, e_h2, e_el, e_out]}  # type: ignore[dict-item]


def simulate_system(sc: Scenario, spec: SystemSpec | None = None, params: ParamSet | None = None) -> SystemResult:
    return SystemModel(sc, spec, params).run()
