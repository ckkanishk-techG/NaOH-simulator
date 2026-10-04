r"""Gas phase: equations of state for H2 and Henry/Sechenov dissolution.

EOS options (H2 only; air and vapour are ideal, Dalton mixing):

* ``ideal``:       :math:`P = nRT/V`
* ``abel_noble``:  :math:`P = nRT/(V-nb)`
* ``peng_robinson``: :math:`P=\frac{RT}{v-b}-\frac{a\alpha}{v^2+2bv-b^2}`
"""

from __future__ import annotations

import math

from ..constants import R
from . import propdb as P

_B_AN = P.get("abel_noble_b")
_TC, _PC, _W = P.get("pr_Tc_H2"), P.get("pr_Pc_H2"), P.get("pr_omega_H2")
_PR_A = 0.45724 * R * R * _TC * _TC / _PC  # magic:  (Peng-Robinson constants)
_PR_B = 0.07780 * R * _TC / _PC  # magic:
_PR_K = 0.37464 + 1.54226 * _W - 0.26992 * _W * _W  # magic:
_KH25 = P.get("henry_H2")
_KH_D = P.get("henry_H2_dlnk_d_invT")
_KS = P.get("sechenov_H2_NaOH")
EOS_NAMES = ("ideal", "abel_noble", "peng_robinson")


def pressure_h2(n: float, V: float, T: float, eos: str = "ideal") -> float:
    """Partial pressure of ``n`` mol H2 occupying ``V`` m3 at ``T`` [Pa]."""
    if n <= 0.0:
        return 0.0
    if eos == "ideal":
        return n * R * T / V
    if eos == "abel_noble":
        return n * R * T / max(V - n * _B_AN, 1.0e-12)
    if eos == "peng_robinson":
        v = V / n
        tr = T / _TC
        alpha = (1.0 + _PR_K * (1.0 - math.sqrt(tr))) ** 2
        return R * T / max(v - _PR_B, 1.0e-9) - _PR_A * alpha / (v * v + 2.0 * _PR_B * v - _PR_B**2)
    raise ValueError(f"unknown EOS {eos!r}")


def compressibility_h2(n: float, V: float, T: float, eos: str) -> float:
    """Z = P V / (n R T) for the chosen EOS."""
    return pressure_h2(n, V, T, eos) * V / (n * R * T)


def henry_h2(T: float, c_naoh: float = 0.0) -> float:
    """Henry solubility of H2 [mol m-3 Pa-1] with van't Hoff T-dependence and salting out."""
    k = _KH25 * math.exp(_KH_D * (1.0 / T - 1.0 / 298.15))  # magic:
    return k * 10.0 ** (-_KS * c_naoh)
