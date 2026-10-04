r"""Level-1 lumped model (also the shared engine for L2 size classes).

State (SI): per-class remaining volume fractions :math:`f_i`, moles of Na+, Al(OH)4-, gibbsite,
liquid water, dissolved H2, gas H2, air, water vapour, temperatures of contents ``T`` and wall
``Tw``, film activity, bubble coverage, crystal moments and cumulative ledgers.

Per mole Al reacted: Al + OH- + 3 H2O -> Al(OH)4- + 3/2 H2.

.. math:: \dot f_i = -\frac{j_i A_{0,i} f_i^{g}}{n_{0,i}},\qquad R=\sum_i j_i A_{0,i} f_i^g,\qquad
          \dot n_{AlO}=R-r_p,\quad \dot n_{W}=-3R-e

Energy (species-enthalpy form; outflows carry their own enthalpy so they cancel):

.. math:: \Big(\sum_i n_i c_{p,i}\Big)\dot T=-Q_{lw}-Q_{cool}-\sum_i h_i(T)\,\dot n_i^{rx+tr}
          -\sum_{dose}\dot n_i\,[h_i(T)-h_i(T_{in})]

Wall:  :math:`C_w \dot T_w = Q_{lw}-Q_{amb}`.  Valve opening/closing is an ODE event (hysteresis).
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from ..constants import (
    BAR,
    G_ACC,
    KELVIN_OFFSET,
    LITRE,
    ML,
    MW_AIR,
    MW_ALOH4,
    MW_H2,
    MW_H2O,
    MW_NA,
    P_ATM,
    SIGMA_SB,
    T_REF,
    T_STP,
    F,
    R,
)
from ..thermo import electrolyte, gas, pitzer, solubility, species, water
from ..thermo import propdb as DB
from ..thermo.ranges import collect
from .geometry import Bins, area_fraction, make_bins
from .params import ParamSet
from .rates import alloy_multiplier, damkohler, make_rate_model
from .scenario import Scenario

SP = ("Al(s)", "H2O(l)", "H2O(g)", "OH-(aq)", "Na+(aq)", "Al(OH)4-(aq)", "Al(OH)3(gibbsite)",
      "H2(g)", "H2(aq)", "air(g)")
_CP = np.array([species.cp(s) for s in SP])
_HF = np.array([DB.get(f"hf:{s}") for s in SP])
(I_AL, I_WL, I_WG, I_OH, I_NA, I_ALO, I_GIB, I_H2, I_H2D, I_AIR) = range(10)
MW_OH = 17.007e-3  # kg/mol  # magic:  (O + H, from constants would be circular-free; see test)

CUM = ("E_amb", "E_cool", "E_outh", "E_in", "Q_rxn", "gen_H2", "vent_H2", "vent_air", "vent_v",
       "stack_H2", "purge_H2", "purge_air", "purge_v", "dose_w", "dose_na", "room_H2", "creep",
       "t_starve", "boil_mol", "evap_mol", "I_demand_t", "I_deliv_t")
_SCALARS = ("nNa", "nAlO", "nGib", "nW", "nH2d", "nH2", "nAir", "nV", "T", "Tw", "film", "thb",
            "m0", "m1", "m2")
_MOM_SCALE = (1.0e9, 1.0e3, 1.0e-3)  # magic:  numerical scaling of crystal moments


@dataclass
class SolverSettings:
    rtol: float = 1.0e-7
    atol: float = 1.0e-10
    method: str = "LSODA"
    dt_out: float = 1.0
    max_step: float = 60.0


@dataclass
class SimResult:
    t: np.ndarray
    series: dict[str, np.ndarray]
    summary: dict[str, float]
    ledger: dict[str, float]
    diagnostics: dict[str, Any]
    scenario: Scenario | None = None
    y: np.ndarray | None = None
    model: Any = None
    events: dict[str, float] = field(default_factory=dict)

    def __getitem__(self, k: str) -> np.ndarray:
        return self.series[k]


def interp_step(profile: list[tuple[float, float]], t: float) -> float:
    """Step-hold lookup of a (time, value) profile; 0 before the first point."""
    if not profile:
        return 0.0
    i = bisect.bisect_right([a for a, _ in profile], t) - 1
    return profile[i][1] if i >= 0 else 0.0


def _sigmoid(x: float) -> float:
    if x < -50.0:  # magic:
        return 0.0
    if x > 50.0:  # magic:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


class L1Model:
    """Lumped model; with ``Scenario.psd`` it carries several size classes (L2)."""

    def __init__(self, sc: Scenario, params: ParamSet | None = None,
                 settings: SolverSettings | None = None, extras: Any = None) -> None:
        self.sc = sc
        self.p = params or ParamSet()
        self.st = settings or SolverSettings()
        self.extras = extras
        self.bins: Bins = make_bins(sc)
        self.nb = self.bins.nb
        self.mult = alloy_multiplier(sc.alloy, sc.activator_ppm, self.p)
        self.rate = make_rate_model(sc.rate_model, self.p, self.mult)
        self.idx = {k: self.nb + i for i, k in enumerate(_SCALARS)}
        self.cidx = {k: self.nb + len(_SCALARS) + i for i, k in enumerate(CUM)}
        self.ny = self.nb + len(_SCALARS) + len(CUM)
        self.r_ves = self.p["r_ves"]
        self.area_xs = math.pi * self.r_ves**2
        self.V_ves = sc.v_vessel_mL * ML
        self.H_ves = self.V_ves / self.area_xs
        r_o = self.r_ves + self.p["t_ves"]
        self.A_out = 2.0 * math.pi * r_o * self.H_ves + 2.0 * math.pi * r_o**2
        self.C_wall = self.p["m_ves"] * self.p["cp_hdpe"]
        self.T_amb = sc.T_amb_C + KELVIN_OFFSET
        self.f_s = DB.get("vanish_frac")
        self.kmt0 = self.p["k_mt"]
        self.valve_open = sc.mode == "open"
        self.Av_relief = self.p["valve_Cv"] * DB.get("valve_Av_per_Cv")
        self.Av_open = math.pi * (sc.vent_diameter_mm * 1.0e-3) ** 2 / 4.0
        self.P_set = P_ATM + sc.p_relief_bar_g * BAR
        self.P_reseat = P_ATM + (1.0 - self.p["valve_hyst"]) * sc.p_relief_bar_g * BAR
        self.gamma = DB.get("gas_gamma")
        self.dp_smooth = DB.get("valve_smooth_dP")
        self.u: tuple[float, float] = (0.0, 0.0)  # (water dose mol/s, NaOH conc. of dose mol/L)
        self.rho_w = DB.get("rho_water_25")
        self.n_al_total0 = float(self.bins.n0.sum())
        self.events = {}
        self.n_valve_events = 0

    # ------------------------------------------------------------------ initial state
    def initial_state(self) -> np.ndarray:
        sc, ix = self.sc, self.idx
        y = np.zeros(self.ny)
        y[: self.nb] = 1.0
        T0 = sc.T0_C + KELVIN_OFFSET
        V_l = sc.v_liq_mL * ML
        y[ix["nNa"]] = sc.c_naoh_M * V_l / LITRE
        rho = electrolyte.density(sc.c_naoh_M, T0)
        m_w = rho * V_l - y[ix["nNa"]] * 39.997e-3  # magic:  (NaOH molar mass)
        y[ix["nW"]] = m_w / MW_H2O
        y[ix["T"]] = y[ix["Tw"]] = T0
        y[ix["film"]] = self.rate.film_initial()
        Vg = self.V_ves - V_l - float(self.bins.v0.sum())
        aw = pitzer.water_activity(y[ix["nNa"]] / m_w, 0.0, T0, model=sc.activity_model)
        pv = aw * water.psat(T0)
        y[ix["nV"]] = pv * Vg / (R * T0)
        y[ix["nAir"]] = max(P_ATM - pv, 0.0) * Vg / (R * T0)
        return y

    # ------------------------------------------------------------------ helpers
    def liquid(self, y: np.ndarray) -> tuple[float, float, float]:
        """(V_l [m3], c_OH [mol/L], c_Al [mol/L]) from state."""
        ix = self.idx
        nNa, nAlO, nW, T = y[ix["nNa"]], max(y[ix["nAlO"]], 0.0), y[ix["nW"]], y[ix["T"]]
        nOH = max(nNa - nAlO, 0.0)
        m_sol = nW * MW_H2O + nNa * MW_NA + nOH * MW_OH + nAlO * MW_ALOH4
        rho, _ = electrolyte.density_from_mass(nNa, m_sol, T)
        V_l = m_sol / rho
        return V_l, nOH / (V_l / LITRE), nAlO / (V_l / LITRE)

    def gas_volume(self, y: np.ndarray, V_l: float) -> float:
        f = np.maximum(y[: self.nb], 0.0)
        v_s = float((f * self.bins.v0).sum()) + max(y[self.idx["nGib"]], 0.0) * DB.get("vm_gibbsite")
        return max(self.V_ves - V_l - v_s, 1.0e-6)

    def pressure(self, y: np.ndarray) -> float:
        ix = self.idx
        T = y[ix["T"]]
        V_l, _, _ = self.liquid(y)
        Vg = self.gas_volume(y, V_l)
        return (gas.pressure_h2(max(y[ix["nH2"]], 0.0), Vg, T, self.sc.eos)
                + (max(y[ix["nAir"]], 0.0) + max(y[ix["nV"]], 0.0)) * R * T / Vg)

    def set_rate_context(self, T: float, nOH: float, nAlO: float, kg_w: float, V_l: float, c_oh: float) -> None:
        """Give activity-based rate models the surface activities (Pitzer or ideal)."""
        rate = self.rate
        if not getattr(rate, "needs_activity", False):
            return
        m_oh, m_al = nOH / kg_w, nAlO / kg_w
        if getattr(rate, "use_act", False):
            _, ln_oh, ln_al = pitzer.ln_gamma(m_oh, m_al, T, model=self.sc.activity_model)
            rate.ctx = (m_oh * math.exp(ln_oh), m_al * math.exp(ln_al))
        else:
            rate.ctx = (c_oh, nAlO / (V_l / LITRE))

    def orifice_mol(self, area: float, P: float, T: float, M: float) -> float:
        r"""Compressible orifice molar flow [mol/s] to ambient: :math:`\dot n=C_d A P\psi/\sqrt{RTM}`."""
        if P <= P_ATM:
            return 0.0
        g, cd, pr = self.gamma, self.p["valve_Cd"], P_ATM / P
        crit = (2.0 / (g + 1.0)) ** (g / (g - 1.0))
        if pr <= crit:
            psi = math.sqrt(g) * (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))
        else:
            psi = math.sqrt(2.0 * g / (g - 1.0) * (pr ** (2.0 / g) - pr ** ((g + 1.0) / g)))
        dp = P - P_ATM
        smooth = dp / (dp + self.dp_smooth)  # ~ dp^1.5 near zero: finite Jacobian
        return cd * area * P * psi * smooth / math.sqrt(R * T * M)

    # ------------------------------------------------------------------ RHS
    def make_rhs(self) -> Callable[[float, np.ndarray], np.ndarray]:  # noqa: C901
        sc, p, ix, cx, nb = self.sc, self.p, self.idx, self.cidx, self.nb
        n0, a0, g_exp, d_m = self.bins.n0, self.bins.a0, self.bins.g, self.bins.d_m
        T_amb, f_s, rate, extras, me = self.T_amb, self.f_s, self.rate, self.extras, self
        evap_k, boil_k, kla = p["evap_k"], p["boil_k"], p["kla_H2"]
        cool_UA, T_cool = sc.cooling_UA_W_K, sc.coolant_T_C + KELVIN_OFFSET
        k_hd, t_w, emis, k_liq, beta = p["k_hdpe"], p["t_ves"], p["emis_hdpe"], p["k_liq"], p["rho_beta_T"]
        stir_mult = p["stirred_h_mult"] if sc.stirred else 1.0
        h_forced = p["h_out_forced"] if sc.forced_convection else 0.0
        dpmin, dpw, purge_flow, h2min = p["reg_dP_min"], p["reg_width"], p["purge_flow"], p["h2_min_frac"]
        mode, cells, load, demand_prof = sc.mode, sc.cells, sc.load_steps, sc.demand_h2_mol_s
        precip, am, eos = sc.precipitation, sc.activity_model, sc.eos
        r_ves, A_xs, H_ves, A_out = self.r_ves, self.area_xs, self.H_ves, self.A_out
        kv, vm_g, cp_m = DB.get("kv_crystal"), DB.get("vm_gibbsite"), DB.get("cp_liq_mass")
        a_nu, a_al, a_pr, a_k = DB.get("air_nu"), DB.get("air_alpha"), DB.get("air_Pr"), DB.get("air_k")
        creep_a, creep_n, creep_q = p["creep_A"], p["creep_n"], p["creep_Q"]
        creep_ref, creep_t = DB.get("creep_sref"), DB.get("hdpe_T_creep_ref")
        hf, cp = _HF, _CP
        rho_w = self.rho_w
        ms = _MOM_SCALE
        room_ach = sc.room_ach / 3600.0
        has_valve = mode != "sealed"
        adiabatic = sc.adiabatic
        needs_act = getattr(rate, "needs_activity", False)
        use_act = getattr(rate, "use_act", False)

        def rhs(t: float, y: np.ndarray) -> np.ndarray:
            dy = np.zeros_like(y)
            f = np.maximum(y[:nb], 0.0)
            nNa, nAlO, nGib = y[ix["nNa"]], max(y[ix["nAlO"]], 0.0), max(y[ix["nGib"]], 0.0)
            nW, nH2d, nH2 = max(y[ix["nW"]], 1.0e-9), max(y[ix["nH2d"]], 0.0), max(y[ix["nH2"]], 0.0)
            nAir, nV = max(y[ix["nAir"]], 0.0), max(y[ix["nV"]], 0.0)
            T, Tw, film, thb = y[ix["T"]], y[ix["Tw"]], y[ix["film"]], y[ix["thb"]]
            nOH = max(nNa - nAlO, 0.0)
            V_l, c_oh, _ = me.liquid(y)
            Vg = me.gas_volume(y, V_l)
            kg_w = nW * MW_H2O
            aw = pitzer.water_activity(nOH / kg_w, nAlO / kg_w, T, model=am)
            P_h2 = gas.pressure_h2(nH2, Vg, T, eos)
            pv = nV * R * T / Vg
            P_tot = P_h2 + (nAir + nV) * R * T / Vg
            # ---- surface reaction
            kmt = extras.kmt(T, c_oh, d_m, me) if extras is not None else me.kmt0
            if needs_act:
                m_oh_, m_al_ = nOH / kg_w, nAlO / kg_w
                if use_act:
                    _, ln_oh_, ln_al_ = pitzer.ln_gamma(m_oh_, m_al_, T, model=am)
                    rate.ctx = (m_oh_ * math.exp(ln_oh_), m_al_ * math.exp(ln_al_))
                else:
                    rate.ctx = (c_oh, nAlO / (V_l / LITRE))
            j, _cs = rate.flux(T, c_oh, film, thb if extras is not None else 0.0, kmt)
            af = area_fraction(f, g_exp, f_s)
            r_i = np.broadcast_to(np.asarray(j, float), (nb,)) * a0 * af
            R_al = float(r_i.sum())
            dy[:nb] = -r_i / n0
            # ---- precipitation (off by default)
            r_p = 0.0
            if precip:
                m_oh, m_al = nOH / kg_w, nAlO / kg_w
                S = solubility.supersaturation(m_oh, m_al, T, sc.polymorph, am)
                J, G = solubility.nucleation_rate(S, T), solubility.growth_rate(S, T)
                Ls = min(solubility.critical_size(S, T), 1.0e-5) if S > 1.0 else 0.0  # magic:
                mu0, mu1, mu2 = y[ix["m0"]] * ms[0], y[ix["m1"]] * ms[1], y[ix["m2"]] * ms[2]
                dy[ix["m0"]] = J / ms[0]
                dy[ix["m1"]] = (G * mu0 + J * Ls) / ms[1]
                dy[ix["m2"]] = (2.0 * G * mu1 + J * Ls**2) / ms[2]
                r_p = min(V_l / vm_g * kv * (3.0 * G * mu2 + J * Ls**3), nAlO / 10.0 + 1.0e-12)  # magic:
            # ---- H2 transfer, evaporation, boiling
            q_h2 = kla * (nH2d - gas.henry_h2(T, c_oh) * P_h2 * V_l)
            e_evap = evap_k * A_xs * (aw * water.psat(T) - pv) / (R * T)
            tb = water.tboil(max(P_tot, 1.0e3) / aw)  # magic:
            e_boil = boil_k / MW_H2O * max(T - tb, 0.0)
            e_tot = e_evap + e_boil
            # ---- dosing
            u_w, c_dose = me.u
            u_na = c_dose * (u_w * MW_H2O / rho_w) / LITRE  # mol NaOH/s
            # ---- gas outflows
            n_gas = nH2 + nAir + nV
            xv3 = np.array([nH2, nAir, nV]) / n_gas if n_gas > 0 else np.array([0.0, 1.0, 0.0])
            M_mix = xv3[0] * MW_H2 + xv3[1] * MW_AIR + xv3[2] * MW_H2O
            vent = 0.0
            if mode == "open":
                vent = me.orifice_mol(me.Av_open, P_tot, T, M_mix)
            elif has_valve and me.valve_open:
                vent = me.orifice_mol(me.Av_relief, P_tot, T, M_mix)
            stack = purge = I_dem = I_del = 0.0
            if mode in ("regulated", "demand"):
                if mode == "regulated":
                    I_dem = interp_step(load, t)
                    dem = cells * I_dem / (2.0 * F)
                else:
                    dem = interp_step(demand_prof, t)
                if dem > 0.0:
                    gp = _sigmoid((P_tot - P_ATM - dpmin) / dpw)
                    gy = _sigmoid((nH2 / max(nH2 + nAir, 1.0e-30) - h2min) / 0.01)  # magic:
                    stack, purge = dem * gp * gy, purge_flow * gp * (1.0 - gy)
                    I_del = I_dem * stack / dem
            o = vent * xv3 + purge * xv3
            o[0] += stack
            # ---- species rates (reaction + transfer) and dosing
            rt = np.zeros(10)
            rt[I_AL], rt[I_OH], rt[I_WL] = -R_al, -R_al + r_p, -3.0 * R_al - e_tot
            rt[I_ALO], rt[I_GIB], rt[I_H2D], rt[I_H2], rt[I_WG] = R_al - r_p, r_p, 1.5 * R_al - q_h2, q_h2, e_tot
            dy[ix["nNa"]] = u_na
            dy[ix["nAlO"]] = R_al - r_p
            dy[ix["nGib"]] = r_p
            dy[ix["nW"]] = -3.0 * R_al - e_tot + u_w
            dy[ix["nH2d"]] = 1.5 * R_al - q_h2
            dy[ix["nH2"]] = q_h2 - o[0]
            dy[ix["nAir"]] = -o[1]
            dy[ix["nV"]] = e_tot - o[2]
            # ---- energy
            h = hf + cp * (T - T_REF)
            n_sp = np.array([float((f * n0).sum()), nW, nV, nOH, nNa, nAlO, nGib, nH2, nH2d, nAir])
            C_tot = float(n_sp @ cp)
            h_rt = float(h @ rt)
            h_dose = 0.0
            if u_w > 0.0:
                dT_in = T - T_amb
                h_dose = (u_w * cp[I_WL] + u_na * (cp[I_NA] + cp[I_OH])) * dT_in
            h_l = V_l / A_xs
            A_in = 2.0 * math.pi * r_ves * h_l + A_xs
            rho_l = electrolyte.density(c_oh, T)
            nu_l = electrolyte.viscosity(c_oh, T) / rho_l
            al_l = k_liq / (rho_l * cp_m)
            ra = G_ACC * beta * max(abs(T - Tw), 0.1) * h_l**3 / (nu_l * al_l)  # magic:
            nu_in = (0.825 + 0.387 * ra ** (1.0 / 6.0) / (1.0 + (0.492 * al_l / nu_l) ** (9.0 / 16.0)) ** (8.0 / 27.0)) ** 2  # magic:
            h_in = nu_in * k_liq / h_l * stir_mult
            R_in = 1.0 / (h_in * A_in) + t_w / (2.0 * k_hd * A_in)
            Q_lw = (T - Tw) / R_in
            Q_cool = cool_UA * (T - T_cool)
            ra_o = G_ACC / (0.5 * (Tw + T_amb)) * max(abs(Tw - T_amb), 0.1) * H_ves**3 / (a_nu * a_al)  # magic:
            nu_o = (0.825 + 0.387 * ra_o ** (1.0 / 6.0) / (1.0 + (0.492 / a_pr) ** (9.0 / 16.0)) ** (8.0 / 27.0)) ** 2  # magic:
            h_o = max(nu_o * a_k / H_ves, h_forced)
            Q_amb = (Tw - T_amb) * h_o * A_out / (1.0 + h_o * t_w / (2.0 * k_hd))
            Q_amb += emis * SIGMA_SB * A_out * (Tw**4 - T_amb**4)
            if adiabatic:
                Q_amb = 0.0
            dy[ix["T"]] = (-Q_lw - Q_cool - h_rt - h_dose) / C_tot
            dy[ix["Tw"]] = (Q_lw - Q_amb) / me.C_wall
            dy[ix["film"]] = rate.film_rate(film, T, c_oh)
            if extras is not None:
                dy[ix["thb"]] = extras.coverage_rate(thb, T, c_oh, np.broadcast_to(np.asarray(j, float), (nb,)),
                                                     a0 * af, V_l, me)
            # ---- ledgers
            dy[cx["E_amb"]], dy[cx["E_cool"]] = Q_amb, Q_cool
            dy[cx["E_outh"]] = float(o @ np.array([h[I_H2], h[I_AIR], h[I_WG]]))
            h_in_dose = u_w * (hf[I_WL] + cp[I_WL] * (T_amb - T_REF)) + u_na * (
                hf[I_NA] + cp[I_NA] * (T_amb - T_REF) + hf[I_OH] + cp[I_OH] * (T_amb - T_REF))
            dy[cx["E_in"]] = h_in_dose
            dy[cx["Q_rxn"]] = -R_al * (-h[I_AL] - h[I_OH] - 3.0 * h[I_WL] + h[I_ALO] + 1.5 * h[I_H2D])
            dy[cx["gen_H2"]] = 1.5 * R_al
            dy[cx["vent_H2"]], dy[cx["vent_air"]], dy[cx["vent_v"]] = vent * xv3
            dy[cx["stack_H2"]] = stack
            dy[cx["purge_H2"]], dy[cx["purge_air"]], dy[cx["purge_v"]] = purge * xv3
            dy[cx["dose_w"]], dy[cx["dose_na"]] = u_w, u_na
            dy[cx["room_H2"]] = (vent + purge) * xv3[0] - room_ach * y[cx["room_H2"]]
            sig = max(P_tot - P_ATM, 0.0) * r_ves / t_w
            dy[cx["creep"]] = (creep_a * (sig / creep_ref) ** creep_n
                               * math.exp(-creep_q / R * (1.0 / Tw - 1.0 / creep_t)))
            dy[cx["boil_mol"]], dy[cx["evap_mol"]] = e_boil, e_evap
            dy[cx["I_demand_t"]], dy[cx["I_deliv_t"]] = I_dem, I_del
            dy[cx["t_starve"]] = 1.0 if (I_dem > 0 and I_del < 0.98 * I_dem) else 0.0  # magic:
            return dy

        return rhs

    # ------------------------------------------------------------------ integration
    def _breakpoints(self, t0: float, t1: float) -> list[float]:
        pts = {t for t, _ in self.sc.load_steps + self.sc.demand_h2_mol_s if t0 < t < t1}
        return sorted(pts)

    def integrate(self, y: np.ndarray, t0: float, t1: float, t_eval: np.ndarray | None = None
                  ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate from t0 to t1 (valve events and profile breakpoints handled).

        Returns (t array, y array [ny, nt]); the last column is the state at ``t1``."""
        rhs = self.make_rhs()
        ts_out: list[np.ndarray] = [np.array([t0])]
        ys_out: list[np.ndarray] = [y.reshape(-1, 1)]
        cuts = [t0] + self._breakpoints(t0, t1) + [t1]
        has_valve = self.sc.mode not in ("sealed", "open")
        pg_set, pg_rs = self.P_set, self.P_reseat
        t, yc = t0, y.copy()
        for a, b in zip(cuts[:-1], cuts[1:], strict=True):
            t = max(t, a)
            guard = 0
            while t < b - 1.0e-12:  # magic:
                guard += 1
                if guard > 500:  # magic:
                    raise RuntimeError("too many valve events")
                events = None
                if has_valve:
                    if self.valve_open:
                        def ev(_t: float, yy: np.ndarray) -> float:
                            return self.pressure(yy) - pg_rs
                        ev.direction = -1.0  # type: ignore[attr-defined]
                    else:
                        def ev(_t: float, yy: np.ndarray) -> float:  # type: ignore[misc]
                            return self.pressure(yy) - pg_set
                        ev.direction = 1.0  # type: ignore[attr-defined]
                    ev.terminal = True  # type: ignore[attr-defined]
                    events = ev
                te = None
                if t_eval is not None:
                    te = t_eval[(t_eval > t) & (t_eval <= b)]
                    if te.size == 0:
                        te = np.array([b])
                    elif te[-1] < b:
                        te = np.append(te, b)
                sol = solve_ivp(rhs, (t, b), yc, method=self.st.method, t_eval=te, events=events,
                                rtol=self.st.rtol, atol=self.st.atol, max_step=self.st.max_step)
                if not sol.success:
                    raise RuntimeError(f"integration failed: {sol.message}")
                if np.size(sol.t):
                    ts_out.append(np.asarray(sol.t))
                    ys_out.append(np.asarray(sol.y))
                if sol.status == 1 and sol.t_events[0].size:  # valve event
                    t = float(sol.t_events[0][0])
                    yc = sol.y_events[0][0].copy()
                    self.valve_open = not self.valve_open
                    self.events.setdefault("t_valve_open" if self.valve_open else "t_valve_close", t)
                    self.n_valve_events += 1
                    if t_eval is None:
                        ts_out.append(np.array([t]))
                        ys_out.append(yc.reshape(-1, 1))
                else:
                    t = b
                    yc = sol.y[:, -1].copy()
        return np.concatenate(ts_out), np.concatenate(ys_out, axis=1)

    events: dict[str, float]
    n_valve_events: int = 0

    def run(self, y0: np.ndarray | None = None) -> SimResult:
        """Full simulation over ``Scenario.duration_s``."""
        self.events = {}
        self.n_valve_events = 0
        self.valve_open = self.sc.mode == "open"
        y = self.initial_state() if y0 is None else y0
        t_grid = np.arange(0.0, self.sc.duration_s + 1.0e-9, self.st.dt_out)
        with collect() as viol:
            t, ys = self.integrate(y, 0.0, self.sc.duration_s, t_grid)
            # keep only the requested grid (event points removed), dedupe
            order = np.argsort(t, kind="stable")
            t, ys = t[order], ys[:, order]
            sel = np.isin(np.round(t, 9), np.round(t_grid, 9))
            _, uniq = np.unique(np.round(t[sel], 9), return_index=True)
            t, ys = t[sel][uniq], ys[:, sel][:, uniq]
            series = self.observables(t, ys)
        res = SimResult(t=t, series=series, summary={}, ledger={}, diagnostics={}, scenario=self.sc,
                        y=ys, model=self, events=dict(self.events))
        res.diagnostics["range_violations"] = [str(v) for v in viol.values()]
        res.diagnostics["n_valve_events"] = self.n_valve_events
        res.ledger = self.ledger(ys)
        res.summary = self.summarize(res)
        return res

    # ------------------------------------------------------------------ outputs
    def observables(self, t: np.ndarray, ys: np.ndarray) -> dict[str, np.ndarray]:
        ix, cx, sc, p = self.idx, self.cidx, self.sc, self.p
        n = len(t)
        keys = ("T", "Tw", "P", "V_l", "c_oh", "c_al", "Vg", "al_g", "flow", "Da", "sf", "aw", "h2_frac",
                "S", "vapor_frac", "E_corr", "i_corr")
        out = {k: np.zeros(n) for k in keys}
        rate, extras = self.rate, self.extras
        for i in range(n):
            y = ys[:, i]
            f = np.maximum(y[: self.nb], 0.0)
            T = y[ix["T"]]
            V_l, c_oh, c_al = self.liquid(y)
            Vg = self.gas_volume(y, V_l)
            P_tot = self.pressure(y)
            nW = max(y[ix["nW"]], 1.0e-9)
            nAlO, nOH = max(y[ix["nAlO"]], 0.0), max(y[ix["nNa"]] - max(y[ix["nAlO"]], 0.0), 0.0)
            aw = pitzer.water_activity(nOH / (nW * MW_H2O), nAlO / (nW * MW_H2O), T, model=sc.activity_model)
            kmt = extras.kmt(T, c_oh, self.bins.d_m, self) if extras is not None else self.kmt0
            self.set_rate_context(T, nOH, nAlO, nW * MW_H2O, V_l, c_oh)
            j, _ = rate.flux(T, c_oh, y[ix["film"]], y[ix["thb"]] if extras is not None else 0.0, kmt)
            r = float((np.broadcast_to(np.asarray(j, float), (self.nb,)) * self.bins.a0
                       * area_fraction(f, self.bins.g, self.f_s)).sum())
            out["T"][i], out["Tw"][i], out["P"][i] = T, y[ix["Tw"]], P_tot
            out["V_l"][i], out["c_oh"][i], out["c_al"][i], out["Vg"][i] = V_l, c_oh, c_al, Vg
            out["al_g"][i] = float((f * self.bins.n0).sum()) * 26.9815385  # magic:  (g/mol)
            out["flow"][i] = 1.5 * r
            out["aw"][i] = aw
            nh, na_, nv = max(y[ix["nH2"]], 0.0), max(y[ix["nAir"]], 0.0), max(y[ix["nV"]], 0.0)
            out["h2_frac"][i] = nh / max(nh + na_ + nv, 1.0e-30)  # magic:
            out["vapor_frac"][i] = nv / max(nh + na_ + nv, 1.0e-30)  # magic:
            if hasattr(rate, "last"):
                out["E_corr"][i] = rate.last().get("E_corr", 0.0)
                out["i_corr"][i] = rate.last().get("i_corr", 0.0)
            kmt_s = float(np.mean(kmt))
            out["Da"][i] = damkohler(rate.kr(T), getattr(rate, "n", 1.0), c_oh, kmt_s)
            pg = max(P_tot - P_ATM, 0.0)
            sig = pg * self.r_ves / p["t_ves"]
            sy = max(p["sy23"] - p["dsy_dT"] * (y[ix["Tw"]] - DB.get("hdpe_T_creep_ref")), 0.1e6)  # magic:
            out["sf"][i] = 99.0 if pg < 1.0 else min(sy / sig, 99.0)  # magic:
            if sc.precipitation or nAlO > 0:
                out["S"][i] = solubility.supersaturation(nOH / (nW * MW_H2O), nAlO / (nW * MW_H2O), T,
                                                         sc.polymorph, sc.activity_model)
        n_mol = ys[cx["gen_H2"]]
        out["h2_gen_mol"] = n_mol
        out["h2_stp_L"] = n_mol * R * T_STP / P_ATM * 1.0e3
        out["h2_actual_L"] = n_mol * R * out["T"] / np.maximum(out["P"], 1.0) * 1.0e3
        out["h2_max_mol"] = np.full(n, 1.5 * self.n_al_total0)
        for k in ("vent_H2", "stack_H2", "purge_H2", "room_H2", "t_starve", "creep", "E_amb",
                  "E_outh", "Q_rxn", "boil_mol", "evap_mol", "I_demand_t", "I_deliv_t"):
            out[k] = ys[cx[k]]
        out["alu_mol"] = ys[ix["nAlO"]]
        out["nH2_gas"] = ys[ix["nH2"]]
        out["nAir"] = ys[ix["nAir"]]
        out["film"] = ys[ix["film"]]
        out["theta_b"] = ys[ix["thb"]]
        lfl = DB.get("lfl_h2")
        out["room_pct"] = out["room_H2"] * R * 298.15 / P_ATM / sc.room_volume_m3 * 100.0  # magic:
        del lfl
        out["conversion"] = 1.0 - out["al_g"] / (sc.al_mass_g)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["I_ratio"] = np.where(out["I_demand_t"] > 0, 1.0, 1.0)
        return out

    def inventories(self, y: np.ndarray) -> dict[str, float]:
        """Atom inventories (mol) including everything that has left the vessel."""
        ix, cx = self.idx, self.cidx
        f = np.maximum(y[: self.nb], 0.0)
        nAl = float((f * self.bins.n0).sum())
        nAlO, nGib, nW, nV = y[ix["nAlO"]], y[ix["nGib"]], y[ix["nW"]], y[ix["nV"]]
        nOH = y[ix["nNa"]] - nAlO
        nH2g, nH2d = y[ix["nH2"]], y[ix["nH2d"]]
        out_v = y[cx["vent_v"]] + y[cx["purge_v"]]
        out_h2 = y[cx["vent_H2"]] + y[cx["stack_H2"]] + y[cx["purge_H2"]]
        return {
            "Al": nAl + nAlO + nGib,
            "Na": y[ix["nNa"]],
            "O": nW + nV + nOH + 4.0 * nAlO + 3.0 * nGib + out_v,
            "H": 2.0 * (nW + nV) + nOH + 4.0 * nAlO + 3.0 * nGib + 2.0 * (nH2g + nH2d) + 2.0 * out_v + 2.0 * out_h2,
            "charge": y[ix["nNa"]] - nOH - nAlO,
        }

    def ledger(self, ys: np.ndarray) -> dict[str, float]:
        """Conservation residuals (relative) between final and initial state."""
        cx = self.cidx
        y0, y1 = ys[:, 0], ys[:, -1]
        i0, i1 = self.inventories(y0), self.inventories(y1)
        dosed_na = y1[cx["dose_na"]] - y0[cx["dose_na"]]
        dosed_w = y1[cx["dose_w"]] - y0[cx["dose_w"]]
        led = {
            "Al": (i1["Al"] - i0["Al"]) / i0["Al"],
            "Na": (i1["Na"] - i0["Na"] - dosed_na) / i0["Na"],
            "O": (i1["O"] - i0["O"] - dosed_w - dosed_na) / i0["O"],
            "H": (i1["H"] - i0["H"] - 2.0 * dosed_w - dosed_na) / i0["H"],
            "charge": i1["charge"] - i0["charge"],
        }
        # energy: contents + wall enthalpy change must equal -losses - outflow enthalpy + dosing enthalpy
        def enthalpy(y: np.ndarray) -> float:
            ix = self.idx
            T = y[ix["T"]]
            f = np.maximum(y[: self.nb], 0.0)
            n_sp = np.array([float((f * self.bins.n0).sum()), y[ix["nW"]], y[ix["nV"]], y[ix["nNa"]] - y[ix["nAlO"]],
                             y[ix["nNa"]], y[ix["nAlO"]], y[ix["nGib"]], y[ix["nH2"]], y[ix["nH2d"]], y[ix["nAir"]]])
            return float(n_sp @ (_HF + _CP * (T - T_REF))) + self.C_wall * (y[ix["Tw"]] - T_REF)

        dH = enthalpy(y1) - enthalpy(y0)
        flux = (-(y1[cx["E_amb"]] - y0[cx["E_amb"]]) - (y1[cx["E_cool"]] - y0[cx["E_cool"]])
                + (y1[cx["E_in"]] - y0[cx["E_in"]]) - (y1[cx["E_outh"]] - y0[cx["E_outh"]]))
        scale = max(abs(y1[cx["Q_rxn"]]), 1.0)
        led["energy"] = (dH - flux) / scale
        return led

    def summarize(self, res: SimResult) -> dict[str, float]:
        s, t = res.series, res.t
        gen = float(s["h2_gen_mol"][-1])
        cum = s["h2_gen_mol"]
        t90 = float(np.interp(0.9 * gen, cum, t)) if gen > 0 else float("nan")  # magic:
        return {
            "h2_total_mol": gen,
            "h2_total_stp_L": float(s["h2_stp_L"][-1]),
            "h2_theoretical_mol": 1.5 * self.n_al_total0,
            "conversion_pct": 100.0 * gen / (1.5 * self.n_al_total0),
            "peak_flow_mol_s": float(s["flow"].max()),
            "peak_T_C": float(s["T"].max() - KELVIN_OFFSET),
            "peak_Tw_C": float(s["Tw"].max() - KELVIN_OFFSET),
            "peak_P_bar_g": float((s["P"].max() - P_ATM) / BAR),
            "t90_s": t90,
            "min_sf": float(s["sf"].min()),
            "peak_room_pct": float(s["room_pct"].max()),
            "starve_s": float(s["t_starve"][-1]),
            "vent_pct": 100.0 * float(s["vent_H2"][-1]) / gen if gen > 0 else 0.0,
            "stack_h2_mol": float(s["stack_H2"][-1]),
            "peak_Da": float(s["Da"].max()),
            "max_creep_strain": float(s["creep"][-1]),
        }


def simulate(sc: Scenario, params: ParamSet | None = None, settings: SolverSettings | None = None,
             extras: Any = None) -> SimResult:
    """Convenience wrapper: build the model and run it."""
    return L1Model(sc, params, settings, extras).run()
