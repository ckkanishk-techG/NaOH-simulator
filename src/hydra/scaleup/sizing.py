r"""Scale-up: lab vessel -> portable unit -> ATV system. **Screening-level estimates only** - not a design.

Wheel power (steady cruise):  :math:`P = [C_{rr} m g\cos\theta + m g\sin\theta + \tfrac12\rho C_dA v^2]\,v/\eta_{dt} + P_{aux}`.
Hydrogen: :math:`n_{H_2}=E_{el}/(LHV\,\eta_{fc})`, aluminium :math:`n_{Al}=n_{H_2}/(1.5\,y)`; reaction heat
:math:`-\Delta H_r(T)\dot n_{Al}` sets heat-rejection duty and exchanger size; reactor volume from the NaOH charge,
aluminate saturation (no precipitation) and the Al bulk volume; mass/volume budget against a battery pack and a
petrol tank delivering the same range.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

from ..constants import G_ACC, KELVIN_OFFSET, MW_AL, MW_ALOH3, MW_ALOH4, MW_NAOH
from ..core.params import ParamSet
from ..fuelcell.stack import Stack, StackSpec
from ..thermo import propdb as DB
from ..thermo import solubility, species
from ..thermo.ranges import collect

LABEL = "SCREENING-LEVEL ESTIMATE: not a design, not validated against a vehicle; prices/factors are user-entered placeholders."


@dataclass
class VehicleSpec:
    mass_kg: float = 0.0
    CdA: float = 0.0
    crr: float = 0.0
    eta_drive: float = 0.0
    aux_W: float = -1.0
    speed_kmh: float = 30.0
    grade_pct: float = 0.0
    range_km: float = 50.0

    def filled(self, p: ParamSet) -> VehicleSpec:
        return VehicleSpec(self.mass_kg or p["atv_mass_kg"], self.CdA or p["atv_CdA"], self.crr or p["atv_crr"],
                           self.eta_drive or p["atv_eta_drive"], self.aux_W if self.aux_W >= 0 else p["atv_aux_W"],
                           self.speed_kmh, self.grade_pct, self.range_km)


def wheel_power_W(v: VehicleSpec, p: ParamSet | None = None) -> float:
    """Electrical power the drive must receive at steady cruise [W] (includes drivetrain loss and auxiliaries)."""
    p = p or ParamSet()
    v = v.filled(p)
    speed = v.speed_kmh / 3.6  # magic: km/h -> m/s
    theta = math.atan(v.grade_pct / 100.0)
    f_roll = v.crr * v.mass_kg * G_ACC * math.cos(theta)
    f_grade = v.mass_kg * G_ACC * math.sin(theta)
    f_drag = 0.5 * p["rho_air"] * v.CdA * speed**2
    return (f_roll + f_grade + f_drag) * speed / v.eta_drive + v.aux_W


def energy_per_km_Wh(v: VehicleSpec, p: ParamSet | None = None) -> float:
    v = v.filled(p or ParamSet())
    return wheel_power_W(v, p) / v.speed_kmh


def fuel_cell_system_efficiency(p: ParamSet, n_cells: int = 4, i_density_A_cm2: float = 0.3, eta_dcdc: float | None = None) -> dict[str, float]:
    """Stack voltage efficiency at the design current density (LHV basis) times DC/DC and H2 utilisation."""
    st = Stack(StackSpec(n_series=n_cells))
    area = st.cell.area
    i = i_density_A_cm2 * 1e4 * area  # magic: A/cm2 -> A/m2
    v_cell = st.voltage(i) / n_cells
    from ..fuelcell.bop import DCDC

    dc = DCDC(p)
    p_out = st.voltage(i) * i
    eta_dc = eta_dcdc if eta_dcdc is not None else dc.efficiency(max(dc.p_out(p_out), 1e-9))
    eta_v = v_cell / (DB.get("lhv_h2") / (2.0 * 96485.33212))  # magic: F
    return {"v_cell": v_cell, "eta_stack_lhv": eta_v, "eta_dcdc": eta_dc, "eta_util": p["fc_utilization"],
            "eta_total": eta_v * eta_dc * p["fc_utilization"]}


@dataclass
class ScaleUpResult:
    label: str
    cruise_power_W: float
    energy_Wh_per_km: float
    energy_electric_Wh: float
    eta_fc_system: float
    h2_mol: float
    h2_kg: float
    h2_L_stp: float
    al_loaded_kg: float
    naoh_kg: float
    water_kg: float
    liquor_L: float
    reactor_L: float
    Wh_per_g_Al: float
    heat_avg_W: float
    heat_peak_W: float
    exchanger_UA_W_K: float
    exchanger_area_m2: float
    masses_kg: dict[str, float]
    volumes_L: dict[str, float]
    comparison: dict[str, dict[str, float]]
    refill: dict[str, float]
    assumptions: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def size_system(v: VehicleSpec | None = None, p: ParamSet | None = None, n_cells: int = 4, i_density_A_cm2: float = 0.3,
                al_per_cartridge_kg: float | None = None, T_reactor_C: float = 70.0, peak_factor: float = 2.0,
                al_kg_per_cartridge_limit: float = 5.0) -> ScaleUpResult:
    """Size reactor, charge and cooling for ``v.range_km`` at ``v.speed_kmh`` and compare against battery and petrol."""
    p = p or ParamSet()
    v = (v or VehicleSpec()).filled(p)
    t_h = v.range_km / v.speed_kmh
    p_el = wheel_power_W(v, p)
    e_el = p_el * t_h
    eff = fuel_cell_system_efficiency(p, n_cells, i_density_A_cm2)
    n_h2 = e_el * 3600.0 / (DB.get("lhv_h2") * eff["eta_total"])
    n_al = n_h2 / (1.5 * p["al_yield"])
    al_kg = n_al * MW_AL
    naoh_kg = n_al * p["naoh_excess"] * MW_NAOH
    # liquor: water from the practical maximum NaOH molality; aluminate beyond saturation (x margin) precipitates as gibbsite,
    # regenerating NaOH (so the charge is NOT limited by aluminate solubility - but the precipitate must be handled/credited)
    n_na = n_al * p["naoh_excess"]
    water_kg = n_na / p["liquor_max_molality"]
    with collect("ignore"):
        sat = solubility.equilibrium_aluminate(p["liquor_max_molality"], T_reactor_C + KELVIN_OFFSET)
    sat_ratio = sat / p["liquor_max_molality"]  # soluble aluminate per Na at saturation
    precip_frac = max(0.0, 1.0 - p["liquor_aluminate_margin"] * sat_ratio * p["naoh_excess"])
    gibbsite_kg = precip_frac * n_al * MW_ALOH3
    naoh_regen_mol = precip_frac * n_al
    min_excess_no_precip = 1.0 / (p["liquor_aluminate_margin"] * sat_ratio)
    liquor_L = (water_kg + naoh_kg) / 1.1  # magic: ~1.1 kg/L alkaline liquor
    al_vol_L = al_kg / DB.get("rho_al") * 1e3 / p["al_bulk_packing"]  # bulk volume of the Al charge [L]
    reactor_L = (liquor_L + al_vol_L) / (1.0 - p["headspace_fraction"])
    # heat
    dh = -species.reaction_enthalpy(T_reactor_C + KELVIN_OFFSET)  # J per mol Al
    mol_al_s = n_al / (t_h * 3600.0)
    q_avg = dh * mol_al_s
    q_peak = q_avg * peak_factor
    ua = q_peak / p["cooling_dT_K"]
    area = ua / p["cooling_h_eff"]
    # masses
    m_react = reactor_L * p["reactor_wall_mass_per_L"]
    n_cells_needed = max(n_cells, math.ceil(p_el / (i_density_A_cm2 * 25.0 * eff["v_cell"])))  # magic: 25 cm2 cell area
    m_stack = n_cells_needed * p["stack_mass_per_cell"]
    m_bop = p["bop_mass_fraction"] * (m_react + m_stack)
    masses = {"aluminium": al_kg, "NaOH": naoh_kg, "water": water_kg, "reactor": m_react, "stack": m_stack, "bop": m_bop}
    masses["total"] = sum(masses.values())
    volumes = {"reactor": reactor_L, "stack": n_cells_needed * 0.02, "bop": 0.3 * reactor_L}  # magic: ~20 mL per cell, BOP volume rule
    volumes["total"] = sum(volumes.values())
    # comparison at equal range
    e_batt_nom = e_el / p["batt_eta_rt"]
    batt_Wh = e_batt_nom / p["batt_dod"]
    petrol_L = e_el / p["petrol_eta"] / 1e3 / p["petrol_kWh_per_L"]
    comp = {
        "battery": {"Wh_installed": batt_Wh, "mass_kg": batt_Wh / p["batt_Wh_per_kg"], "volume_L": batt_Wh / p["batt_Wh_per_L"]},
        "petrol": {"litres": petrol_L, "mass_kg": petrol_L * p["petrol_kg_per_L"] * (1 + p["petrol_tank_mass_factor"]),
                   "volume_L": petrol_L * 1.2},  # magic: tank volume overhead
        "hydra": {"mass_kg": masses["total"], "volume_L": volumes["total"]},
    }
    comp["hydra"]["mass_vs_battery"] = masses["total"] / comp["battery"]["mass_kg"]
    comp["hydra"]["volume_vs_battery"] = volumes["total"] / comp["battery"]["volume_L"]
    # refill strategy
    cart = al_per_cartridge_kg or min(al_kg, al_kg_per_cartridge_limit)
    n_cart = math.ceil(al_kg / cart - 1e-9)
    refill = {"cartridge_Al_kg": cart, "cartridges_per_range": float(n_cart), "km_per_cartridge": v.range_km / max(al_kg / cart, 1e-12),  # magic: floor
              "swap_minutes_per_range": n_cart * p["refill_minutes"], "spent_liquor_kg_per_range": water_kg + naoh_kg + al_kg * (MW_ALOH4 / MW_AL - 1.0)}
    wh_per_g_al = e_el / (al_kg * 1e3)
    return ScaleUpResult(LABEL, p_el, e_el / v.range_km, e_el, eff["eta_total"], n_h2, n_h2 * 2.016e-3, n_h2 * 22.414, al_kg, naoh_kg,  # magic: g/mol H2, L/mol
                         water_kg, liquor_L, reactor_L, wh_per_g_al, q_avg, q_peak, ua, area, masses, volumes, comp, refill,
                         {"t_hours": t_h, "n_cells": float(n_cells_needed), "precipitate_fraction": precip_frac, "gibbsite_kg": gibbsite_kg,
                          "naoh_regenerated_mol": naoh_regen_mol, "min_naoh_excess_without_precipitation": min_excess_no_precip,
                          "aluminate_sat_ratio": sat_ratio, **eff})
