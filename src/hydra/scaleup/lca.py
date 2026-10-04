r"""Simplified life-cycle CO2e comparison (cradle-to-gate for consumables + use phase). SCREENING LEVEL ONLY.

Al-H2 ATV:   g/km = [m_Al (share primary x EF_prim + share recycled x EF_rec) + m_NaOH EF_NaOH + m_water EF_w
                      - credit x reusable aluminate] / km.
Petrol ATV:  well-to-wheel EF per litre.   Grid EV: kWh/km x grid EF / charging efficiency.
All factors are user-editable placeholders (property DB category ``lca``); results carry low/high ranges from the
plausible ranges of the dominant factors.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..constants import MW_ALOH3
from ..core.params import ParamSet
from ..thermo import propdb as DB
from .sizing import LABEL, ScaleUpResult, VehicleSpec, size_system


@dataclass
class LCAResult:
    label: str
    g_per_km: dict[str, float]
    ranges_g_per_km: dict[str, tuple[float, float]]
    breakdown_al_h2: dict[str, float]
    breakeven_grid_g_per_kWh: float
    recycled_share: float


def lca_compare(sc: ScaleUpResult | None = None, p: ParamSet | None = None, recycled_share: float = 1.0,
                naoh_recovery: float = 0.0, byproduct_reuse: bool = True, v: VehicleSpec | None = None) -> LCAResult:
    """CO2e per km: Al-H2 vs petrol vs grid-charged EV. ``recycled_share`` = fraction of Al from scrap (1 = all scrap)."""
    p = p or ParamSet()
    v = (v or VehicleSpec()).filled(p)
    sc = sc or size_system(v, p)

    def al_h2(ef_prim: float, ef_rec: float, ef_naoh: float, credit: float) -> tuple[float, dict[str, float]]:
        al = sc.al_loaded_kg * (recycled_share * ef_rec + (1 - recycled_share) * ef_prim)
        naoh = sc.naoh_kg * (1.0 - naoh_recovery) * ef_naoh
        water = sc.water_kg * p["ef_water"]
        cr = -(sc.al_loaded_kg * p["al_yield"] * MW_ALOH3 / 26.9815385e-3 * 1.0 * credit) if byproduct_reuse else 0.0  # magic: kg Al(OH)3 per kg Al reacted
        br = {"aluminium": al / v.range_km, "NaOH": naoh / v.range_km, "water": water / v.range_km, "credit": cr / v.range_km}
        return sum(br.values()), br

    nominal, br = al_h2(p["ef_al_primary"], p["ef_al_recycled"], p["ef_naoh"], p["ef_gibbsite_credit"])
    e_km = sc.energy_Wh_per_km / 1e3  # kWh/km at the drive
    grid = e_km / p["batt_eta_rt"] / p["lca_charge_eff"] * p["ef_grid"] * 1e3
    petrol = sc.comparison["petrol"]["litres"] / v.range_km * p["ef_petrol_wtw"] * 1e3
    lo, _ = al_h2(DB.entry("ef_al_primary").min or 0.0, DB.entry("ef_al_recycled").min or 0.0, DB.entry("ef_naoh").min or 0.0,
                  DB.entry("ef_gibbsite_credit").max or 0.0)
    hi, _ = al_h2(DB.entry("ef_al_primary").max or 0.0, DB.entry("ef_al_recycled").max or 0.0, DB.entry("ef_naoh").max or 0.0,
                  DB.entry("ef_gibbsite_credit").min or 0.0)
    ef_lo, ef_hi = DB.entry("ef_grid").min or 0.0, DB.entry("ef_grid").max or 1.0
    g = {"al_h2": nominal * 1e3, "petrol": petrol, "grid_ev": grid}
    rng = {"al_h2": (lo * 1e3, hi * 1e3), "petrol": (petrol * 0.8, petrol * 1.2),  # magic: +-20 % well-to-wheel uncertainty
           "grid_ev": (grid * ef_lo / p["ef_grid"], grid * ef_hi / p["ef_grid"])}
    # grid factor at which the grid EV equals the Al-H2 system [g/kWh]
    be = nominal * 1e3 / (e_km / p["batt_eta_rt"] / p["lca_charge_eff"])
    return LCAResult(LABEL, g, rng, {k: x * 1e3 for k, x in br.items()}, be, recycled_share)
