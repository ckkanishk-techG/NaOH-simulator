r"""Particle geometry and size distributions.

Each size class ``i`` has volume fraction remaining :math:`f_i` (state variable) and surface area

.. math:: A_i = A_{0,i}\, f_i^{g},\quad g=2/3\ (\text{sphere}),\ 1/2\ (\text{wire}),\ 0\ (\text{sheets}).

Sheets (foil, flake, can) dissolve from both faces; edges are neglected (piece edge >> thickness).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..constants import MW_AL
from ..thermo import propdb as P
from .scenario import PSDSpec, Scenario

_SHAPE = {  # form -> (shape class, area exponent g, surface/volume factor numerator)
    "powder": ("sphere", 2.0 / 3.0, 6.0),
    "wire": ("wire", 0.5, 4.0),
    "foil": ("sheet", 0.0, 2.0),
    "flake": ("sheet", 0.0, 2.0),
    "can": ("sheet", 0.0, 2.0),
}


@dataclass
class Bins:
    """Size classes for Al: moles, initial area, exponent, characteristic dimension."""

    d_m: np.ndarray  # characteristic dimension [m]
    n0: np.ndarray  # initial moles Al per class
    a0: np.ndarray  # initial wetted area per class [m2]
    g: float  # area exponent
    v0: np.ndarray  # initial volume per class [m3]
    mass_frac: np.ndarray

    @property
    def nb(self) -> int:
        return len(self.d_m)


def psd_masses(spec: PSDSpec, d_default_um: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (bin dimension [um], mass fractions) for a distribution spec."""
    d50 = spec.d50_um or d_default_um
    if spec.kind == "mono" or spec.n_bins == 1:
        return np.array([d50]), np.array([1.0])
    if spec.kind == "custom":
        if not spec.custom_d_um or not spec.custom_mass_frac:
            raise ValueError("custom PSD needs custom_d_um and custom_mass_frac")
        d = np.asarray(spec.custom_d_um, float)
        w = np.asarray(spec.custom_mass_frac, float)
        return d, w / w.sum()
    nb = spec.n_bins
    if spec.kind == "lognormal":
        s = math.log(spec.sigma_g)
        z = np.linspace(-3.0, 3.0, nb)  # magic:  (+-3 sigma)
        d = d50 * np.exp(s * z)
        w = np.exp(-0.5 * z**2)  # mass-weighted lognormal
    else:  # Rosin-Rammler: mass fraction passing = 1 - exp(-(d/d63)^n)
        d63 = d50 / math.log(2.0) ** (1.0 / spec.n_rr)
        edges = np.linspace(0.0, 3.0 * d63, nb + 1)  # magic:
        cdf = 1.0 - np.exp(-((edges / d63) ** spec.n_rr))
        w = np.diff(cdf)
        d = 0.5 * (edges[1:] + edges[:-1])
        keep = w > 0
        d, w = d[keep], w[keep]
    return d, w / w.sum()


def make_bins(sc: Scenario) -> Bins:
    cls, g, sv_num = _SHAPE[sc.form]
    spec = sc.psd or PSDSpec(kind="mono", n_bins=1)
    d_um, w = psd_masses(spec, sc.dim_um)
    d = d_um * 1.0e-6
    rho = P.get("rho_al")
    m = sc.al_mass_g * 1.0e-3 * w
    v0 = m / rho
    a0 = v0 * sv_num / d
    if sc.form == "can":
        active = P.get("can_active_coated") + (1.0 - P.get("can_active_coated")) * sc.lacquer_removed
        a0 = a0 * active
    return Bins(d_m=d, n0=m / MW_AL, a0=a0, g=g, v0=v0, mass_frac=w)


def area_fraction(f: np.ndarray, g: float, f_s: float) -> np.ndarray:
    """A/A0 with a short ramp so that sheets (g=0) vanish smoothly below ``f_s``."""
    fc = np.maximum(f, 0.0)
    ramp = np.minimum(fc / f_s, 1.0)
    return (fc**g if g > 0 else 1.0) * ramp
