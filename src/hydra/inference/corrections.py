r"""Gas-volume corrections calculator and mass-loss method (spec 20.5).

Convert a raw reading into moles of H2 *produced*, with uncertainty (Monte-Carlo propagation).

* ``water_displacement`` (wet gas over water): pressure in the bottle
  :math:`p = P_{baro} - \rho g h` (h = inside water level above outside level), dry-gas partial pressure
  :math:`p_{H_2}=p-p_{sat}(T_w)`, collected moles :math:`pV/RT`, plus H2 dissolved in the collection water
  :math:`k_H(T_w)\,p_{H_2}V_{w}`.
* ``gas_syringe`` (humid gas at room temperature): :math:`p_{H_2}=P_{baro}-\phi\,p_{sat}(T)`.
* ``flowmeter``: totaliser volume at the meter's reference conditions (default 0 C, 1 atm).
* ``mass_loss``: reactor on a balance, :math:`n_{H_2}=(\Delta m - m_{vap})/M_{H_2}`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..constants import G_ACC, MW_H2, P_ATM, R, T_STP
from ..thermo import gas, water

METHODS = ("water_displacement", "gas_syringe", "flowmeter", "mass_loss")


@dataclass
class Reading:
    method: str
    value: float  # volume [L] (or mass loss [g] for 'mass_loss')
    T_K: float = 295.15
    P_baro_Pa: float = P_ATM
    head_m: float = 0.0  # water_displacement: inside level above outside level
    water_T_K: float | None = None
    contact_water_L: float = 0.0  # water volume in contact with the gas (dissolution)
    rh: float = 1.0  # gas_syringe humidity (relative)
    ref_T_K: float = T_STP  # flowmeter reference temperature
    ref_P_Pa: float = P_ATM
    vapor_loss_g: float = 0.0  # mass_loss: water vapour/mist that left with the gas
    rho_water: float = 998.0
    unc: dict[str, float] = field(default_factory=dict)  # 1-sigma of any numeric field above


def moles_from_reading(r: Reading) -> float:
    """Moles of H2 produced for a single reading (no uncertainty)."""
    if r.method not in METHODS:
        raise ValueError(f"unknown method {r.method!r}")
    if r.method == "mass_loss":
        return (r.value - r.vapor_loss_g) * 1.0e-3 / MW_H2
    v = r.value * 1.0e-3
    if r.method == "water_displacement":
        tw = r.water_T_K if r.water_T_K is not None else r.T_K
        p_in = r.P_baro_Pa - r.rho_water * G_ACC * r.head_m
        p_h2 = p_in - water.psat(tw)
        n = p_h2 * v / (R * r.T_K)
        n += gas.henry_h2(tw) * p_h2 * r.contact_water_L * 1.0e-3
        return float(n)
    if r.method == "gas_syringe":
        p_h2 = r.P_baro_Pa - r.rh * water.psat(r.T_K)
        return float(p_h2 * v / (R * r.T_K))
    return float(r.ref_P_Pa * v / (R * r.ref_T_K))  # flowmeter / totaliser at reference conditions


def moles_with_uncertainty(r: Reading, n_mc: int = 2000, seed: int = 0) -> tuple[float, float]:
    """(mean, sd) of n_H2 [mol] with the input uncertainties in ``r.unc`` propagated by Monte Carlo."""
    rng = np.random.default_rng(seed)
    n0 = moles_from_reading(r)
    if not r.unc:
        return n0, 0.0
    samples = np.empty(n_mc)
    for i in range(n_mc):
        kw = {k: getattr(r, k) + sd * rng.standard_normal() for k, sd in r.unc.items()}
        samples[i] = moles_from_reading(Reading(**{**{f: getattr(r, f) for f in r.__dataclass_fields__ if f != "unc"}, **kw}))
    return float(n0), float(samples.std(ddof=1))


def correction_table(r: Reading) -> dict[str, float]:
    """Breakdown of each correction relative to the naive ideal-gas reading ``P_baro V/(R T)``."""
    naive = r.P_baro_Pa * r.value * 1.0e-3 / (R * r.T_K) if r.method != "mass_loss" else float("nan")
    full = moles_from_reading(r)
    out = {"naive_mol": naive, "corrected_mol": full}
    if r.method == "water_displacement":
        tw = r.water_T_K if r.water_T_K is not None else r.T_K
        out["vapour_pressure_Pa"] = water.psat(tw)
        out["hydrostatic_Pa"] = -r.rho_water * G_ACC * r.head_m
        out["dissolved_mol"] = gas.henry_h2(tw) * (r.P_baro_Pa + out["hydrostatic_Pa"] - out["vapour_pressure_Pa"]) \
            * r.contact_water_L * 1.0e-3
        out["relative_correction"] = full / naive - 1.0
    return out


def mass_loss_estimate_vapor(n_h2: float, T_gas: float, aw: float = 1.0, P: float = P_ATM) -> float:
    """Water vapour carried by n_h2 mol of H2 leaving saturated at T_gas [g] (for the mass-loss method)."""
    pv = aw * water.psat(T_gas)
    return n_h2 * pv / max(P - pv, 1.0) * 18.015  # magic: floor, g/mol water
