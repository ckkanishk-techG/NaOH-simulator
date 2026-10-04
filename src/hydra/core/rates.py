r"""Dissolution-rate models: flux of Al per unit surface area, :math:`j` [mol m-2 s-1].

``arrhenius``:  :math:`j = k_{25} m_{alloy} e^{-E_a/R(1/T-1/T_{25})}\,c^n\,\psi\,(1-\theta_b)`

``mass_transfer``: surface kinetics in series with OH- diffusion through the boundary layer:
:math:`k_r c_s^n = k_{mt}(c_b - c_s)` solved for the surface concentration ``c_s``;
Damkoehler number :math:`Da = k_r c_b^{n-1}/k_{mt}`.
"""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from ..constants import T_REF, R
from .params import ParamSet

_T25 = T_REF
_ALLOY = {"pure": "mult_pure", "1xxx": "mult_1xxx", "3xxx": "mult_3xxx", "6061": "mult_6061"}


def alloy_multiplier(alloy: str, activator_ppm: float, p: ParamSet) -> float:
    return p[_ALLOY[alloy]] * (1.0 + p["activator_gain_per_ppm"] * activator_ppm)


class RateModel(Protocol):
    name: str

    def flux(self, T: float, c: float, film: float, theta_b: float, kmt: np.ndarray | float
             ) -> tuple[np.ndarray | float, np.ndarray | float]:
        """Return (j [mol m-2 s-1], c_surface [mol/L])."""

    def film_rate(self, film: float, T: float, c: float) -> float:
        """d(film activity)/dt."""

    def film_initial(self) -> float: ...

    def kr(self, T: float) -> float:
        """Surface rate coefficient (j at c = 1 mol/L, no film/coverage)."""


def _series_resistance(kr: float, n: float, c: float, km: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve kr cs^n = km (c - cs) for the surface concentration (damped Newton, early exit)."""
    cs = c / (1.0 + kr * max(c, 1.0e-12) ** (n - 1.0) / km)  # magic: floor; exact for n = 1
    for _ in range(12):  # magic: Newton iterations cap
        g = kr * cs**n - km * (c - cs)
        dg = kr * n * np.maximum(cs, 1.0e-12) ** (n - 1.0) + km  # magic: floor
        step = g / dg
        cs = np.minimum(np.maximum(cs - step, 0.0), c)
        if np.max(np.abs(step)) < 1.0e-13 * max(c, 1.0e-12):  # magic: convergence tolerance
            break
    return cs, kr * cs**n


class ArrheniusModel:
    name = "arrhenius"

    def __init__(self, p: ParamSet, mult: float, mass_transfer: bool = False) -> None:
        self.p, self.mult, self.mt = p, mult, mass_transfer
        self.k25, self.ea, self.n = p["k25"], p["Ea"], p["n_oh"]
        self.tau = p["tau_ind"]
        self.name = "mass_transfer" if mass_transfer else "arrhenius"

    def kr(self, T: float) -> float:
        return self.k25 * self.mult * math.exp(-self.ea / R * (1.0 / T - 1.0 / _T25))

    def flux(self, T: float, c: float, film: float, theta_b: float, kmt: np.ndarray | float
             ) -> tuple[np.ndarray | float, np.ndarray | float]:
        kr = self.kr(T) * film * (1.0 - theta_b)
        c = max(c, 0.0)
        if not self.mt:
            return kr * c**self.n, c
        # solve kr*cs^n = kmt*1000*(c-cs)  (kmt in m/s, c in mol/L -> mol/m3 factor 1000)
        km = np.asarray(kmt, float) * 1.0e3
        cs, j = _series_resistance(kr, self.n, c, km)
        return j, cs

    def film_rate(self, film: float, T: float, c: float) -> float:
        return (1.0 - film) / self.tau

    def film_initial(self) -> float:
        return 0.0


def make_rate_model(name: str, p: ParamSet, mult: float, mass_transfer: bool = False) -> RateModel:
    if name == "arrhenius":
        return ArrheniusModel(p, mult, mass_transfer)
    if name == "mass_transfer":
        return ArrheniusModel(p, mult, True)
    if name == "electrochemical":
        from ..electrochem.model import ElectrochemModel

        return ElectrochemModel(p, mult, mass_transfer)
    raise ValueError(f"unknown rate model {name!r}")


def damkohler(kr_c: float, n: float, c: float, kmt: float) -> float:
    """Da = k_r c^(n-1) / k_mt   (c in mol/L, kmt m/s, k_r mol m-2 s-1 per (mol/L)^n)."""
    return kr_c * max(c, 1.0e-6) ** (n - 1.0) / (kmt * 1.0e3)
