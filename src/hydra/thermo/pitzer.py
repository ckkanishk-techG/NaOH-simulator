r"""Pitzer activity model for the Na+ / OH- / Al(OH)4- system (molal scale), with extended
Debye-Hueckel as a fast fallback.

Osmotic coefficient (single cation Na+, anions a):

.. math::
   \phi-1=\frac{2}{\sum m}\Big[-\frac{A_\phi I^{3/2}}{1+b\sqrt I}
   +\sum_a m_{Na}m_a (B^\phi_{Na,a}+Z C_{Na,a})+\sum_{a<a'} m_a m_{a'}\Theta_{aa'}\Big],\;
   B^\phi=\beta_0+\beta_1 e^{-\alpha\sqrt I}

Binary NaOH parameters are valid near 25 C only; use outside [288, 308] K is flagged by the
range guard (temperature dependence of the Pitzer parameters is not modelled).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..constants import MW_H2O
from . import propdb as P
from .ranges import check

_A = P.get("debye_huckel_A_phi")
_B = P.get("pitzer_b")
_ALPHA = P.get("pitzer_alpha")
_T_LO, _T_HI = P.trange("pitzer_beta0_NaOH")
_M_MAX = 6.0  # molal validity limit for the binary NaOH parameter set  # magic:


@dataclass(frozen=True)
class PitzerParams:
    b0_oh: float
    b1_oh: float
    c_oh: float
    b0_al: float
    b1_al: float
    c_al: float
    theta: float

    @classmethod
    def default(cls) -> PitzerParams:
        return cls(P.get("pitzer_beta0_NaOH"), P.get("pitzer_beta1_NaOH"), P.get("pitzer_cphi_NaOH"),
                   P.get("pitzer_beta0_NaAlOH4"), P.get("pitzer_beta1_NaAlOH4"), 0.0,
                   P.get("pitzer_theta_OH_AlOH4"))


def _g(x: float) -> float:
    return 2.0 * (1.0 - (1.0 + x) * math.exp(-x)) / (x * x)


def _gp(x: float) -> float:
    return -2.0 * (1.0 - (1.0 + x + 0.5 * x * x) * math.exp(-x)) / (x * x)


def _guard(m_na: float, T: float) -> None:
    check("pitzer", "T", T, _T_LO, _T_HI)
    check("pitzer", "molality", m_na, 0.0, _M_MAX)


def osmotic_coefficient(m_oh: float, m_al: float = 0.0, T: float = 298.15,
                        pp: PitzerParams | None = None, model: str = "pitzer") -> float:
    """Osmotic coefficient of NaOH + NaAl(OH)4 solution (molalities m_oh, m_al of the anions)."""
    pp = pp or PitzerParams.default()
    m_na = m_oh + m_al
    if m_na <= 0.0:
        return 1.0
    _guard(m_na, T)
    i = m_na
    si = math.sqrt(i)
    dh = -_A * i * si / (1.0 + _B * si)
    if model == "debye_huckel":
        return 1.0 + 2.0 / (m_oh + m_al + m_na) * dh
    e = math.exp(-_ALPHA * si)
    z = 2.0 * m_na
    bphi_oh = pp.b0_oh + pp.b1_oh * e
    bphi_al = pp.b0_al + pp.b1_al * e
    s = dh + m_na * (m_oh * (bphi_oh + z * 0.5 * pp.c_oh) + m_al * (bphi_al + z * 0.5 * pp.c_al))
    s += m_oh * m_al * pp.theta
    return 1.0 + 2.0 / (m_na + m_oh + m_al) * s


def water_activity(m_oh: float, m_al: float = 0.0, T: float = 298.15,
                   pp: PitzerParams | None = None, model: str = "pitzer") -> float:
    r""":math:`\ln a_w = -\phi \sum m_i M_w`."""
    phi = osmotic_coefficient(m_oh, m_al, T, pp, model)
    return math.exp(-phi * (2.0 * m_oh + 2.0 * m_al - m_al) * MW_H2O)  # sum m_i = Na+ + OH- + Al


def ln_gamma(m_oh: float, m_al: float = 0.0, T: float = 298.15,
             pp: PitzerParams | None = None, model: str = "pitzer") -> tuple[float, float, float]:
    """(ln gamma_Na+, ln gamma_OH-, ln gamma_Al(OH)4-), molal scale."""
    pp = pp or PitzerParams.default()
    m_na = m_oh + m_al
    if m_na <= 0.0:
        return 0.0, 0.0, 0.0
    _guard(m_na, T)
    i = m_na
    si = math.sqrt(i)
    f_dh = -_A * (si / (1.0 + _B * si) + 2.0 / _B * math.log(1.0 + _B * si))
    if model == "debye_huckel":
        return f_dh, f_dh, f_dh
    x = _ALPHA * si
    g, gp = _g(x), _gp(x)
    b_oh, b_al = pp.b0_oh + pp.b1_oh * g, pp.b0_al + pp.b1_al * g
    bp_oh, bp_al = pp.b1_oh * gp / i, pp.b1_al * gp / i
    c_oh, c_al = 0.5 * pp.c_oh, 0.5 * pp.c_al
    z = 2.0 * m_na
    f = f_dh + m_na * (m_oh * bp_oh + m_al * bp_al)
    ln_na = (f + m_oh * (2.0 * b_oh + z * c_oh) + m_al * (2.0 * b_al + z * c_al)
             + m_na * (m_oh * c_oh + m_al * c_al))
    ln_oh = (f + m_na * (2.0 * b_oh + z * c_oh) + m_na * (m_oh * c_oh + m_al * c_al)
             + 2.0 * m_al * pp.theta)
    ln_al = (f + m_na * (2.0 * b_al + z * c_al) + m_na * (m_oh * c_oh + m_al * c_al)
             + 2.0 * m_oh * pp.theta)
    return ln_na, ln_oh, ln_al


def mean_gamma_naoh(m: float, T: float = 298.15) -> float:
    """Mean ionic activity coefficient of pure NaOH."""
    ln_na, ln_oh, _ = ln_gamma(m, 0.0, T)
    return math.exp(0.5 * (ln_na + ln_oh))
