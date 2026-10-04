r"""Mixed-potential (corrosion couple) theory for Al in alkaline solution.

Anodic:  Al + 4OH- -> Al(OH)4- + 3e-      :math:`i_a = i_{0a}[e^{\alpha_a f\eta_a}-e^{-\beta_a f\eta_a}]`
Cathodic: 2H2O + 2e- -> H2 + 2OH-         :math:`i_c = i_{0c}[e^{\alpha_c f\eta_c}-e^{-\beta_c f\eta_c}]`

with :math:`f=F/RT`, :math:`\eta_a=E-E_a`, :math:`\eta_c=E_c-E`. Equilibrium potentials (Nernst):

.. math:: E_a=E^0_a+\frac{RT}{3F}\ln\frac{a_{Al(OH)_4^-}}{a_{OH^-}^4},\qquad
          E_c=E^0_c-\frac{RT}{F}\ln a_{OH^-}-\frac{RT}{2F}\ln(1/p_{H_2})

The corrosion potential solves :math:`i_a(E)=i_c(E)(1+r_{galv}\,\cdot)`; the Faraday relation gives
:math:`\dot n_{Al}=i_{corr}A/3F` and :math:`\dot n_{H_2}=i_{corr}A/2F`. In the Tafel limit:

.. math:: \ln i_{corr}=\frac{\alpha_c}{\alpha_a+\alpha_c}\ln i_{0a}+\frac{\alpha_a}{\alpha_a+\alpha_c}\ln i_{0c}
          +f\frac{\alpha_a\alpha_c}{\alpha_a+\alpha_c}(E_c-E_a)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..constants import F, R


@dataclass(frozen=True)
class CouplePar:
    i0a: float
    i0c: float
    alpha_a: float
    beta_a: float
    alpha_c: float
    beta_c: float
    i0c_noble: float = 0.0
    galv_ratio: float = 0.0
    i_passive: float = 0.0


def nernst_potentials(T: float, a_oh: float, a_al: float, e0_a: float, e0_c: float,
                      p_h2: float = 1.0) -> tuple[float, float]:
    """(E_a, E_c) in volts at temperature T, activities a_oh, a_al (molal), H2 pressure [atm]."""
    rt_f = R * T / F
    e_a = e0_a + rt_f / 3.0 * (math.log(a_al) - 4.0 * math.log(a_oh))
    e_c = e0_c - rt_f * math.log(a_oh) + 0.5 * rt_f * math.log(max(p_h2, 1.0e-12))  # magic: floor
    return e_a, e_c


def _terms(e: float, T: float, par: CouplePar, e_a: float, e_c: float, i0a: float, i0c: float,
           i0n: float) -> tuple[float, float, float, float]:
    f = F / (R * T)
    eta_a = e - e_a
    eta_c = e_c - e
    xa1, xa2 = par.alpha_a * f * eta_a, -par.beta_a * f * eta_a
    xc1, xc2 = par.alpha_c * f * eta_c, -par.beta_c * f * eta_c
    clip = 80.0  # magic: exp overflow guard
    xa1, xa2, xc1, xc2 = (max(min(x, clip), -clip) for x in (xa1, xa2, xc1, xc2))
    ia = i0a * (math.exp(xa1) - math.exp(xa2)) + par.i_passive
    dia = i0a * f * (par.alpha_a * math.exp(xa1) + par.beta_a * math.exp(xa2))
    k = 1.0 + par.galv_ratio * (i0n / i0c if i0c > 0 else 0.0)
    ic = k * i0c * (math.exp(xc1) - math.exp(xc2))
    dic = -k * i0c * f * (par.alpha_c * math.exp(xc1) + par.beta_c * math.exp(xc2))
    return ia, dia, ic, dic


def solve_corrosion(T: float, par: CouplePar, e_a: float, e_c: float, i0a: float, i0c: float,
                    i0n: float = 0.0, tol: float = 1.0e-12) -> tuple[float, float]:
    """Solve the mixed-potential condition. Returns (E_corr [V], i_corr [A m-2] on the Al area).

    Safeguarded Newton on ``h(E)=i_a(E)-i_c(E)``, bracketed by [E_a, E_c] (h<0 at E_a, h>0 at E_c)."""
    lo, hi = e_a, e_c
    if hi <= lo:  # no driving force: no net corrosion
        return 0.5 * (lo + hi), 0.0
    f = F / (R * T)
    # Tafel analytic guess
    aa, ac = par.alpha_a, par.alpha_c
    guess = (math.log(max(i0c, 1e-300) / max(i0a, 1e-300)) / f + aa * e_a + ac * e_c) / (aa + ac)  # magic: floor
    e = min(max(guess, lo), hi)
    for _ in range(100):  # magic: iteration cap
        ia, dia, ic, dic = _terms(e, T, par, e_a, e_c, i0a, i0c, i0n)
        h = ia - ic
        if h > 0:
            hi = e
        else:
            lo = e
        dh = dia - dic
        e_new = e - h / dh if dh > 0 else 0.5 * (lo + hi)
        if not (lo < e_new < hi):
            e_new = 0.5 * (lo + hi)
        if abs(e_new - e) < tol:
            e = e_new
            break
        e = e_new
    ia, _, _, _ = _terms(e, T, par, e_a, e_c, i0a, i0c, i0n)
    return e, max(ia, 0.0)


def tafel_icorr(T: float, par: CouplePar, e_a: float, e_c: float, i0a: float, i0c: float) -> float:
    """Closed-form corrosion current density in the Tafel limit (no back reactions, no galvanic)."""
    f = F / (R * T)
    aa, ac = par.alpha_a, par.alpha_c
    s = aa + ac
    ln_i = (ac / s) * math.log(i0a) + (aa / s) * math.log(i0c) + f * aa * ac / s * (e_c - e_a)
    return math.exp(ln_i)
