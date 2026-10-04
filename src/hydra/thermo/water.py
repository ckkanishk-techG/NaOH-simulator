r"""Water and steam properties.

Vapour pressure by the Antoine equation (two ranges), :math:`\log_{10} P_{mmHg} = A - B/(C+T_C)`.
"""

from __future__ import annotations

import math

from ..constants import KELVIN_OFFSET, MMHG, T_REF
from . import propdb as P
from .ranges import check

_LO = (P.get("antoine_lo_A"), P.get("antoine_lo_B"), P.get("antoine_lo_C"))
_HI = (P.get("antoine_hi_A"), P.get("antoine_hi_B"), P.get("antoine_hi_C"))
_T_SPLIT = P.trange("antoine_hi_A")[0]
_T_TOP = P.trange("antoine_hi_A")[1]
_T_BOT = P.trange("antoine_lo_A")[0]
_TC = P.get("water_Tc")
_WATSON = P.get("watson_exp")
_HVAP_REF = P.get("hvap_ref")
_T_REF = T_REF


def psat(T: float) -> float:
    """Saturation pressure of pure water [Pa] at temperature ``T`` [K]."""
    check("water.psat", "T", T, _T_BOT, _T_TOP)
    a, b, c = _LO if T < _T_SPLIT else _HI
    return float(MMHG * 10.0 ** (a - b / (c + T - KELVIN_OFFSET)))


def tboil(p: float) -> float:
    """Boiling temperature [K] of pure water at pressure ``p`` [Pa] (inverse Antoine)."""
    lp = math.log10(p / MMHG)
    a, b, c = _LO
    tc = b / (a - lp) - c
    if tc + KELVIN_OFFSET >= _T_SPLIT:
        a, b, c = _HI
        tc = b / (a - lp) - c
    return tc + KELVIN_OFFSET


def hvap(T: float) -> float:
    """Latent heat of vaporisation of water [J mol-1] (Watson correlation)."""
    check("water.hvap", "T", T, _T_BOT, _T_TOP)
    return _HVAP_REF * (max(_TC - T, 0.0) / (_TC - _T_REF)) ** _WATSON
