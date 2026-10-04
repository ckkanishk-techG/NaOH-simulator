r"""Safety engine: pre-run screening, run-time checks and time-to-limit forecasts.

**This is a predictive model, not a safety certification.** Physical experiments still need
supervision, venting, pressure relief and protective equipment that work independently of any software.

Checks: overpressure (relief set point / hoop safety factor / creep), HDPE temperature, thermal
runaway (dT/dt, d2T/dt2, Semenov number :math:`\psi = Q_{rx}E_a/(R T^2 UA)`), room H2 against the
4-75 % flammable range, NaOH corrosivity, boiling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from ..constants import BAR, KELVIN_OFFSET, MW_AL, P_ATM, T_REF, R
from ..thermo import propdb as DB
from ..thermo import species, water
from .l1 import SimResult
from .params import ParamSet
from .scenario import Scenario

DISCLAIMER = (
    "Predictive model, NOT a safety certification. Experiments with Al/NaOH/H2 require supervision, "
    "ventilation, a physical pressure-relief device and protective equipment (goggles, gloves, "
    "apron) that work independently of this software. Never rely on model output to clear a hazard."
)


@dataclass
class Flag:
    level: str  # 'info' | 'warn' | 'danger'
    code: str
    message: str
    value: float | None = None
    limit: float | None = None


@dataclass
class SafetyReport:
    flags: list[Flag] = field(default_factory=list)
    forecasts: dict[str, float] = field(default_factory=dict)
    disclaimer: str = DISCLAIMER

    @property
    def level(self) -> str:
        order = {"info": 0, "warn": 1, "danger": 2}
        return max((f.level for f in self.flags), key=lambda x: order[x], default="info")

    @property
    def unsafe(self) -> bool:
        return self.level == "danger"

    def add(self, level: str, code: str, msg: str, value: float | None = None, limit: float | None = None) -> None:
        self.flags.append(Flag(level, code, msg, value, limit))


def time_to_limit(t: np.ndarray, x: np.ndarray, limit: float, window_s: float = 60.0) -> float:
    """Linear-extrapolation time [s] until ``x`` reaches ``limit`` (inf if not approaching, 0 if past)."""
    if x[-1] >= limit:
        return 0.0
    m = t >= t[-1] - window_s
    if m.sum() < 2:
        return math.inf
    slope = np.polyfit(t[m], x[m], 1)[0]
    return (limit - x[-1]) / slope if slope > 0 else math.inf


def adiabatic_peak_T(sc: Scenario) -> float:
    """Upper-bound temperature [K] if all Al reacts with no heat loss (species enthalpy balance)."""
    n_al = sc.al_mass_g * 1e-3 / MW_AL
    T0 = sc.T0_C + KELVIN_OFFSET
    dh = -species.reaction_enthalpy(T0)
    c_liq = sc.v_liq_mL * 1e-6 * 1.0e3 * DB.get("cp_liq_mass") * 1.08  # magic: (g/mL density, rough)
    c_wall = DB.get("m_ves") * DB.get("cp_hdpe")
    return T0 + dh * n_al / (c_liq + c_wall)


def screen(sc: Scenario, params: ParamSet | None = None) -> SafetyReport:
    """Pre-run what-if screening (cheap, conservative) - flags unsafe scenarios before simulating."""
    p = params or ParamSet()
    rep = SafetyReport()
    n_al = sc.al_mass_g * 1e-3 / MW_AL
    n_h2 = 1.5 * n_al
    T_peak = adiabatic_peak_T(sc)
    t_max = p["T_hdpe_max"]
    if T_peak > t_max:
        rep.add("danger" if T_peak > t_max + 20 else "warn", "ADIABATIC_T",  # magic:
                f"No-loss temperature rise reaches {T_peak - KELVIN_OFFSET:.0f} C (> HDPE service limit "
                f"{t_max - KELVIN_OFFSET:.0f} C); actual peak depends on cooling and reaction speed.", T_peak, t_max)
    t_boil = water.tboil(P_ATM)
    if T_peak > t_boil:  # magic:
        rep.add("warn", "BOILING", f"Boiling likely (no-loss peak {T_peak - KELVIN_OFFSET:.0f} C): splatter/mist and steam load.",
                T_peak, t_boil)
    vg = (sc.v_vessel_mL - sc.v_liq_mL) * 1e-6
    if sc.mode == "sealed":
        p_end = n_h2 * R * (sc.T0_C + KELVIN_OFFSET) / vg
        rep.add("danger", "SEALED_NO_RELIEF",
                f"Sealed vessel without relief: {p_end / BAR:.1f} bar of H2 if all Al reacts.", p_end / BAR, 0.0)
    if sc.mode != "open":
        sy = p["sy23"]
        sigma_set = sc.p_relief_bar_g * BAR * p["r_ves"] / p["t_ves"]
        sf = sy / sigma_set
        if sf < DB.get("safety_sf_min"):
            rep.add("danger", "RELIEF_TOO_HIGH", f"Hoop safety factor at relief set point is {sf:.1f} (<"
                    f"{DB.get('safety_sf_min'):.0f}).", sf, DB.get("safety_sf_min"))
    # flammable inventory released to room
    room_n = sc.room_volume_m3 * P_ATM / (R * 298.15)  # magic:
    frac = n_h2 / (room_n + n_h2)
    if frac > DB.get("lfl_h2"):
        rep.add("danger", "ROOM_LFL", f"Venting all H2 into the room gives {100 * frac:.1f}% H2 (LFL "
                f"{100 * DB.get('lfl_h2'):.0f}%) without ventilation.", frac, DB.get("lfl_h2"))
    elif frac > 0.25 * DB.get("lfl_h2"):  # magic:
        rep.add("warn", "ROOM_H2", f"Venting all H2 gives {100 * frac:.2f}% H2 in the room (25% of LFL) - ventilate.",
                frac, 0.25 * DB.get("lfl_h2"))  # magic:
    if sc.c_naoh_M >= DB.get("naoh_corrosive_M"):
        rep.add("warn" if sc.c_naoh_M > 4.0 else "info", "NAOH_CORROSIVE",  # magic:
                f"{sc.c_naoh_M:g} M NaOH is severely corrosive: eye/skin protection, no aluminium-bench spills.",
                sc.c_naoh_M, DB.get("naoh_corrosive_M"))
    rep.add("info", "DISCLAIMER", DISCLAIMER)
    return rep


def semenov_series(res: SimResult, params: ParamSet) -> np.ndarray:
    r"""Semenov number :math:`\psi = \dot Q_{rx} E_a /(R T^2\, UA_{eff})` along the run (nan if undefined)."""
    t = res.t
    s = res.series
    q_rx = np.gradient(s["Q_rxn"], t)
    q_loss = np.gradient(s["E_amb"], t)
    dT = s["T"] - (res.scenario.T_amb_C + KELVIN_OFFSET if res.scenario else T_REF)
    with np.errstate(invalid="ignore", divide="ignore"):
        ua = np.where(dT > 1.0, q_loss / dT, np.nan)
        return q_rx * params["Ea"] / (R * s["T"] ** 2 * ua)


def analyze(res: SimResult, params: ParamSet | None = None) -> SafetyReport:
    """Run-time checks on a result with time-to-limit forecasts evaluated at the end of the run."""
    p = params or ParamSet()
    sc = res.scenario
    assert sc is not None
    s, t = res.series, res.t
    rep = SafetyReport()
    pg = s["P"] - P_ATM
    if sc.mode != "open":
        p_set = sc.p_relief_bar_g * BAR
        if pg.max() > 1.05 * p_set and sc.mode != "sealed":  # magic:
            rep.add("danger", "OVERPRESSURE", f"Peak {pg.max() / BAR:.2f} bar(g) exceeds relief set {p_set / BAR:.2f}.",
                    pg.max() / BAR, p_set / BAR)
        elif pg.max() > DB.get("safety_P_margin") * p_set:
            rep.add("warn", "NEAR_RELIEF", f"Pressure reached {100 * pg.max() / p_set:.0f}% of the relief set point.",
                    pg.max() / BAR, p_set / BAR)
    if s["sf"].min() < DB.get("safety_sf_min"):
        rep.add("danger" if s["sf"].min() < 1.5 else "warn", "HOOP_SF",  # magic:
                f"Minimum hoop safety factor {s['sf'].min():.2f}.", float(s["sf"].min()), DB.get("safety_sf_min"))
    t_hd = p["T_hdpe_max"]
    if s["Tw"].max() > t_hd:
        rep.add("danger", "HDPE_T", f"Wall reached {s['Tw'].max() - KELVIN_OFFSET:.0f} C (> {t_hd - KELVIN_OFFSET:.0f} C service limit).",
                float(s["Tw"].max()), t_hd)
    if s["creep"][-1] > DB.get("creep_strain_limit"):
        rep.add("danger", "CREEP", f"Creep strain {s['creep'][-1]:.3f} exceeds limit.", float(s["creep"][-1]),
                DB.get("creep_strain_limit"))
    if s["boil_mol"][-1] > 0:
        rep.add("warn", "BOILING", "Electrolyte boiled: mist/steam carry-over expected.", float(s["boil_mol"][-1]))
    psi = semenov_series(res, p)
    if np.nanmax(psi, initial=0.0) > DB.get("semenov_psi_crit"):
        rep.add("danger", "SEMENOV", f"Semenov number reached {np.nanmax(psi):.2f} (>1: thermal runaway regime).",
                float(np.nanmax(psi)), DB.get("semenov_psi_crit"))
    dTdt = np.gradient(s["T"], t)
    d2 = np.gradient(dTdt, t)
    rep.forecasts["max_dTdt_K_s"] = float(dTdt.max())
    rep.forecasts["max_d2Tdt2_K_s2"] = float(d2.max())
    if dTdt.max() > 0.5:  # magic:  (K/s)
        rep.add("warn", "FAST_HEATING", f"Heating rate up to {dTdt.max():.2f} K/s.", float(dTdt.max()), 0.5)  # magic:
    room = s["room_pct"] / 100.0
    if room.max() > DB.get("lfl_h2"):
        rep.add("danger", "ROOM_LFL", f"Room H2 reached {100 * room.max():.1f}% (flammable from "
                f"{100 * DB.get('lfl_h2'):.0f}%).", float(room.max()), DB.get("lfl_h2"))
    # in-vessel flammability: H2/air mixtures between LFL and UFL (dry basis)
    h2f = s["nH2_gas"] / np.maximum(s["nH2_gas"] + s["nAir"], 1e-30)  # magic:
    if np.any((h2f > DB.get("lfl_h2")) & (h2f < DB.get("ufl_h2")) & (s["nH2_gas"] > 1e-6)):  # magic:
        rep.add("info", "HEADSPACE_FLAMMABLE", "Headspace passes through the 4-75% H2-in-air flammable range "
                "(unavoidable when starting with air): no ignition sources.")
    # forecasts at the end of the run
    rep.forecasts["t_to_hdpe_limit_s"] = time_to_limit(t, s["Tw"], t_hd)
    if sc.mode not in ("open",):
        rep.forecasts["t_to_relief_s"] = time_to_limit(t, pg, sc.p_relief_bar_g * BAR)
    rep.forecasts["t_to_lfl_room_s"] = time_to_limit(t, room, DB.get("lfl_h2"))
    return rep
