r"""Anion-exchange-membrane fuel cell (H2/air), NiMo HOR anode, Ag ORR cathode, nickel-foam electrodes.

.. math:: V = E_{rev}(T,p) - \eta_{a} - \eta_{c} - i\,\mathrm{ASR}(T,\lambda,x_{CO_3}) - \eta_{conc}

* :math:`E_{rev} = -\Delta G(T)/2F + \frac{RT}{2F}\ln\frac{p_{H_2}p_{O_2}^{1/2}}{a_{H_2O}}` with
  :math:`\Delta G(T)` from the species thermochemistry database (H2 + 1/2 O2 -> H2O(l)).
* Activation: Butler-Volmer solved for each electrode (separate NiMo and Ag parameters).
* Ohmic: ASR scaled by Arrhenius in T, membrane water content :math:`(\lambda_{ref}/\lambda)^m`,
  carbonate fraction and ageing.
* Concentration: :math:`\eta_{conc} = -\frac{RT}{2F}\,c\,\ln(1-i/i_L)`; flooding reduces :math:`i_L`.
* Water (AEM): water is produced at the anode (net 0.5 H2O per electron overall), consumed at the
  cathode; lumped membrane balance :math:`\Gamma\dot\lambda = 0.5\,i/F - N_{evap,c} - N_{evap,a}`.
* Carbonation (optional): :math:`\dot x = k_c\,p_{CO_2}(1-x) - k_r\,i\,x`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..constants import P_ATM, F, R
from ..core.params import ParamSet
from ..thermo import species, water


def e_rev(T: float, p_h2_atm: float, p_o2_atm: float, a_w: float = 1.0) -> float:
    """Reversible cell voltage [V] (Nernst, Gibbs from species data)."""
    dh = species.h("H2O(l)", T) - species.h("H2(g)", T) - 0.5 * species.h("O2(g)", T)
    ds = species.s("H2O(l)", T) - species.s("H2(g)", T) - 0.5 * species.s("O2(g)", T)
    e0 = -(dh - T * ds) / (2.0 * F)
    return e0 + R * T / (2.0 * F) * math.log(max(p_h2_atm, 1.0e-9) * max(p_o2_atm, 1.0e-9) ** 0.5 / a_w)  # magic: floors


def bv_overpotential(i: float, i0: float, alpha: float, beta: float, T: float) -> float:
    """Overpotential [V] solving i = i0 [exp(alpha f eta) - exp(-beta f eta)] (i >= 0)."""
    if i <= 0.0:
        return 0.0
    f = F / (R * T)
    eta = math.asinh(i / (2.0 * i0)) / (f * 0.5 * (alpha + beta))
    for _ in range(40):  # magic: Newton iterations
        x1, x2 = min(alpha * f * eta, 60.0), min(-beta * f * eta, 60.0)  # magic: exp guard
        g = i0 * (math.exp(x1) - math.exp(x2)) - i
        dg = i0 * f * (alpha * math.exp(x1) + beta * math.exp(x2))
        step = g / dg
        eta -= step
        if abs(step) < 1.0e-12:  # magic: tolerance
            break
    return eta


@dataclass
class CellCond:
    """Operating/ageing state of one cell."""

    T: float
    lam: float
    x_carb: float = 0.0
    age_cat: float = 1.0  # multiplies exchange currents (<=1)
    age_mem: float = 1.0  # multiplies ASR (>=1)
    asr_mult: float = 1.0  # manufacturing spread
    iL_scale: float = 1.0  # flow-distribution factor


class AEMCell:
    def __init__(self, params: ParamSet | None = None, area_cm2: float = 25.0) -> None:
        self.p = params or ParamSet()
        self.area = area_cm2 * 1.0e-4

    # --- helpers --------------------------------------------------------------------------------
    def sorption(self, lam: float) -> float:
        """Membrane water activity from water content (power-law isotherm)."""
        p = self.p
        x = max(lam, 0.0) / p["fc_lambda_max"]
        return min(x ** (1.0 / p["fc_sorption_b"]), 1.5)  # magic: >1 = liquid water present

    def flood_risk(self, lam: float) -> float:
        p = self.p
        return min(max((lam - p["fc_lambda_flood"]) / max(p["fc_lambda_max"] - p["fc_lambda_flood"], 1e-9), 0.0), 1.0)  # magic: floor

    def dry_risk(self, lam: float) -> float:
        p = self.p
        return min(max((p["fc_lambda_dry"] - lam) / p["fc_lambda_dry"], 0.0), 1.0)

    def asr(self, c: CellCond) -> float:
        """Area-specific resistance [ohm m2]."""
        p = self.p
        arr = math.exp(p["fc_Ea_asr"] / R * (1.0 / c.T - 1.0 / p["fc_T_ref"]))
        hyd = (p["fc_lambda_ref"] / max(c.lam, 0.3)) ** p["fc_lambda_exp"]  # magic: lambda floor
        carb = 1.0 / (1.0 - c.x_carb * (1.0 - p["fc_sigma_carb_ratio"]))
        return p["fc_asr"] * arr * hyd * carb * c.age_mem * c.asr_mult

    # --- voltage -----------------------------------------------------------------------------------
    def voltage(self, i: float, c: CellCond, p_h2_atm: float = 1.0, p_o2_atm: float | None = None) -> dict[str, float]:
        """Cell voltage at current density ``i`` [A m-2] with loss breakdown."""
        p = self.p
        if p_o2_atm is None:
            p_o2_atm = p["fc_x_o2"] * p["fc_P_cathode"]
        T = c.T
        a_w = max(self.sorption(c.lam), 0.05)  # magic: floor
        e = e_rev(T, p_h2_atm, p_o2_atm, a_w)
        x = 1.0 / T - 1.0 / p["fc_T_ref"]
        i0a = p["fc_i0_hor"] * math.exp(-p["fc_Ea_hor"] / R * x) * max(p_h2_atm, 1e-6) ** p["fc_order_h2"] * c.age_cat  # magic: floor
        i0c = (p["fc_i0_orr"] * math.exp(-p["fc_Ea_orr"] / R * x) * (max(p_o2_atm, 1e-6) / 0.21) ** p["fc_order_o2"]  # magic: floor, air ref
               * c.age_cat)
        eta_a = bv_overpotential(i, i0a, p["fc_alpha_hor"], p["fc_beta_hor"], T)
        eta_c = bv_overpotential(i, i0c, p["fc_alpha_orr"], p["fc_beta_orr"], T)
        r_asr = self.asr(c)
        eta_o = i * r_asr
        il = p["fc_iL"] * c.iL_scale * (1.0 - p["fc_flood_iL_penalty"] * self.flood_risk(c.lam)) * (p_o2_atm / 0.21)  # magic: air ref
        xr = i / il
        eta_k = -R * T / (2.0 * F) * p["fc_conc_factor"] * math.log(1.0 - xr) if xr < 0.999 else 1.0  # magic: cap
        v = max(e - eta_a - eta_c - eta_o - eta_k, 0.0)
        return {"V": v, "E": e, "eta_a": eta_a, "eta_c": eta_c, "eta_ohm": eta_o, "eta_conc": eta_k,
                "asr": r_asr, "iL": il}

    def limiting_current(self, c: CellCond) -> float:
        p = self.p
        return p["fc_iL"] * c.iL_scale * (1.0 - p["fc_flood_iL_penalty"] * self.flood_risk(c.lam))

    # --- water / carbonation / ageing --------------------------------------------------------------
    def water_rates(self, i: float, c: CellCond, rh_anode: float, rh_cathode: float,
                    gas_flow_mol_m2s: float | None = None) -> tuple[float, float]:
        """(dlam/dt, net water evaporation flux) [s-1, mol m-2 s-1]."""
        p = self.p
        psat = water.psat(c.T)
        a = self.sorption(c.lam)
        n_c = p["fc_k_evap_c"] * psat * (a - rh_cathode) / (R * c.T)
        n_a = p["fc_k_evap_a"] * psat * (a - rh_anode) / (R * c.T)
        if gas_flow_mol_m2s is not None:  # water carried out by the anode gas cannot exceed its capacity
            cap = gas_flow_mol_m2s * psat / max(P_ATM - psat, 1.0e3)  # magic: floor
            n_a = min(n_a, cap)
        prod = 0.5 * i / F
        dlam = (prod - n_c - n_a) / p["fc_ion_density"]
        return dlam, n_c + n_a

    def carb_rate(self, i: float, c: CellCond) -> float:
        p = self.p
        if p["fc_carb_on"] < 0.5:  # magic: switch threshold
            return 0.0
        pco2 = p["fc_co2_ppm"] * 1.0e-6 * p["fc_P_cathode"]
        return p["fc_k_carb"] * pco2 * (1.0 - c.x_carb) * 1.0e3 - p["fc_k_regen"] * i * c.x_carb  # magic: ppm scaling to atm*1e3
