"""Balance of plant: gas conditioning (condenser / dryer / mist filter), DC/DC converter, battery buffer."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..constants import P_ATM
from ..core.params import ParamSet
from ..thermo import water


@dataclass
class ConditionedGas:
    rh_stack: float  # relative humidity of the anode feed at stack temperature
    water_condensed_mol_s: float
    mist_g_s: float


class Conditioner:
    """Condenser -> optional dryer -> mist filter between vessel and stack.

    Outgoing vapour is condensed down to the saturation pressure at ``cond_T``; a dryer lowers the dew
    point further. Mist (NaOH aerosol) carried by boiling is removed with efficiency ``cond_mist_eff``;
    what leaks through, and the feed RH, are what the AEM stack sees.
    """

    def __init__(self, params: ParamSet, condenser: bool = True, dryer: bool = False) -> None:
        self.p, self.condenser, self.dryer = params, condenser, dryer

    def condition(self, T_gas: float, vapor_frac: float, total_P: float, gas_flow_mol_s: float,
                  boil_mol_s: float, T_stack: float) -> ConditionedGas:
        p = self.p
        p_w = vapor_frac * total_P
        t_dew = T_gas
        if self.condenser:
            p_w = min(p_w, water.psat(p["cond_T"]) * total_P / P_ATM)
            t_dew = min(T_gas, p["cond_T"])
        if self.dryer:
            p_w = min(p_w, water.psat(p["dryer_dew_K"]))
        p_w_pre = vapor_frac * total_P
        cond_mol = max(p_w_pre - p_w, 0.0) / max(total_P, 1.0) * gas_flow_mol_s  # magic: floor
        mist_in = p["mist_g_per_mol_boil"] * boil_mol_s
        mist_out = mist_in * (1.0 - p["cond_mist_eff"]) if self.condenser else mist_in
        rh = min(p_w / water.psat(T_stack), 1.0)
        _ = t_dew
        return ConditionedGas(rh_stack=rh, water_condensed_mol_s=cond_mol, mist_g_s=mist_out)


class DCDC:
    """Converter loss model: P_in = P_out + P0 + a P_out + b P_out^2."""

    def __init__(self, params: ParamSet) -> None:
        self.p = params

    def p_in(self, p_out: float) -> float:
        if p_out <= 0.0:
            return 0.0
        p = self.p
        return p_out + p["dcdc_P0"] + p["dcdc_a"] * p_out + p["dcdc_b"] * p_out**2

    def p_out(self, p_in: float) -> float:
        """Inverse of :meth:`p_in` (closed form of the quadratic)."""
        p = self.p
        c = p["dcdc_P0"]
        a, b = 1.0 + p["dcdc_a"], p["dcdc_b"]
        if p_in <= c:
            return 0.0
        if b <= 0:
            return (p_in - c) / a
        return (-a + math.sqrt(a * a + 4.0 * b * (p_in - c))) / (2.0 * b)  # magic: quadratic formula

    def efficiency(self, p_out: float) -> float:
        return p_out / self.p_in(p_out) if p_out > 0 else 0.0


@dataclass
class Battery:
    capacity_Wh: float = 0.0
    p_max_W: float = 50.0
    soc: float = 0.5
    soc_min: float = 0.1
    soc_max: float = 0.95

    def exchange(self, p_req: float, dt: float, params: ParamSet) -> float:
        """Battery power [W] actually delivered (>0 discharge, <0 charge) for a request ``p_req``;
        SOC and power limits are enforced and losses applied to the stored energy."""
        if self.capacity_Wh <= 0.0:
            return 0.0
        p = max(min(p_req, self.p_max_W), -self.p_max_W)
        e_wh = self.capacity_Wh * self.soc
        if p > 0:  # discharge
            avail = (e_wh - self.soc_min * self.capacity_Wh) * 3600.0 * params["batt_eta_dis"] / dt
            p = min(p, max(avail, 0.0))
            self.soc -= p * dt / 3600.0 / params["batt_eta_dis"] / self.capacity_Wh
        else:  # charge
            room = (self.soc_max * self.capacity_Wh - e_wh) * 3600.0 / params["batt_eta_chg"] / dt
            p = -min(-p, max(room, 0.0))
            self.soc -= p * dt / 3600.0 * params["batt_eta_chg"] / self.capacity_Wh
        return p
