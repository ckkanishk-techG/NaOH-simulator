r"""Aluminate solubility, supersaturation and gibbsite/bayerite precipitation (CNT + growth).

Equilibrium  Al(OH)3(s) + OH- = Al(OH)4-,  :math:`K(T)=\frac{a_{Al(OH)_4^-}}{a_{OH^-}}`,
:math:`\ln K = \ln K_{25} - \frac{\Delta H}{R}(1/T-1/T_{ref})`.
Supersaturation :math:`S = (a_{Al}/a_{OH})/K`. Nucleation (classical theory):
:math:`J = A\exp[-16\pi\sigma^3 v^2/(3 (kT)^3\ln^2 S)]`; growth :math:`G = k_g (S-1)^g`.

All parameters are uncalibrated placeholders (see property DB): precipitation is OFF by default
in the simulator and only used for diagnostics until calibrated.
"""

from __future__ import annotations

import math

from ..constants import K_B, N_A, T_REF, R
from . import pitzer
from . import propdb as P
from .ranges import check

_LNK0 = P.get("sol_lnK0")
_DH = P.get("sol_dH")
_BAY = P.get("sol_polymorph_bayerite_factor")
_SIG = P.get("nuc_sigma")
_A = P.get("nuc_A")
_KG = P.get("growth_k")
_G = P.get("growth_order")
_VM = P.get("vm_gibbsite")
_T_RANGE = (273.15, 373.15)  # magic:  validity of the van't Hoff form


def log_k(T: float, polymorph: str = "gibbsite") -> float:
    """ln K of the dissolution equilibrium (bayerite is more soluble by a fixed factor)."""
    check("solubility", "T", T, *_T_RANGE)
    lk = _LNK0 - _DH / R * (1.0 / T - 1.0 / T_REF)
    return lk + (math.log(_BAY) if polymorph == "bayerite" else 0.0)


def supersaturation(m_oh: float, m_al: float, T: float, polymorph: str = "gibbsite",
                    model: str = "pitzer") -> float:
    """Activity-based supersaturation ratio S (S>1 means thermodynamically able to precipitate)."""
    if m_al <= 0.0 or m_oh <= 0.0:
        return 0.0
    _, ln_oh, ln_al = pitzer.ln_gamma(m_oh, m_al, T, model=model)
    ratio = (m_al / m_oh) * math.exp(ln_al - ln_oh)
    return ratio / math.exp(log_k(T, polymorph))


def equilibrium_aluminate(m_na: float, T: float, polymorph: str = "gibbsite",
                          model: str = "pitzer") -> float:
    """Saturation aluminate molality for total sodium ``m_na`` (solve S(m_Al)=1 by bisection)."""
    lo, hi = 0.0, m_na * (1.0 - 1.0e-9)
    if supersaturation(m_na - hi, hi, T, polymorph, model) < 1.0:
        return hi  # fully saturated only at the physical limit
    for _ in range(60):  # magic:  bisection iterations
        mid = 0.5 * (lo + hi)
        if supersaturation(m_na - mid, mid, T, polymorph, model) > 1.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def nucleation_rate(s: float, T: float) -> float:
    """CNT nucleation rate J [m-3 s-1]."""
    if s <= 1.0:
        return 0.0
    v = _VM / N_A
    b = 16.0 * math.pi * _SIG**3 * v * v / (3.0 * (K_B * T) ** 3)  # magic:  (CNT prefactor 16 pi/3)
    return float(_A * math.exp(-b / math.log(s) ** 2))


def critical_size(s: float, T: float) -> float:
    """Critical nucleus diameter [m]."""
    if s <= 1.0:
        return float("inf")
    return 4.0 * _SIG * (_VM / N_A) / (K_B * T * math.log(s))


def growth_rate(s: float, T: float) -> float:
    """Linear crystal growth rate G [m s-1] with Arrhenius-free placeholder temperature factor."""
    if s <= 1.0:
        return 0.0
    return _KG * (s - 1.0) ** _G
