r"""Mixed-potential rate model (implements the ``RateModel`` interface of ``hydra.core.rates``).

Surface flux :math:`j = m_{alloy}\,\psi\,(1-\theta_b)\, i_{corr}/(3F)` [mol Al m-2 s-1].
Oxide film: bare fraction :math:`\psi=1-\theta`,
:math:`\dot\psi = k_d(a_{OH},T)(1-\psi) - k_p\,\psi` (dissolution of the Al2O3 film competes with
repassivation: induction period at start, stalling at low [OH-] where :math:`\psi_\infty=k_d/(k_d+k_p)`).
Temperature enters through Arrhenius exchange currents and the Nernst/Tafel :math:`f=F/RT`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..constants import T_REF, F, R
from ..core.params import ParamSet
from ..thermo import pitzer
from .mixed_potential import CouplePar, nernst_potentials, solve_corrosion, tafel_icorr


def from_arrhenius(k25: float, ea: float, n: float, p: ParamSet, a_ref: float = 1.0) -> dict[str, float]:
    """Electrochemical parameters whose Tafel-limit corrosion rate equals the empirical Arrhenius law
    ``k25 exp(-Ea/R(1/T-1/298.15)) c^n`` (at a_Al = floor). Used both as the model default and in the
    'reduces to Arrhenius' verification."""
    aa, ac = p["ec_alpha_a"], p["ec_alpha_c"]
    s = aa + ac
    kap = aa * ac / s
    d_e0 = p["ec_E0_HER"] - p["ec_E0_Al"]
    ga = p["ec_gamma_a"]
    gc = ((n - kap / 3.0) * s - ac * ga) / aa
    e_i = ea + F * kap * d_e0
    f = F / (R * T_REF)
    ln_target = math.log(3.0 * F * k25)  # i_corr at c = 1 mol/L
    ln_i0c = math.log(p["ec_i0c"])
    ln_ref = math.log(a_ref)
    nernst = f * kap * (d_e0 + (R * T_REF / (3.0 * F)) * (ln_ref - math.log(p["ec_aAl_floor"])))
    ln_i0a = (ln_target - (aa / s) * (ln_i0c + gc * ln_ref) - nernst) / (ac / s) - ga * ln_ref
    return {"ec_i0a": math.exp(ln_i0a), "ec_gamma_c": gc, "ec_Ea_a": e_i, "ec_Ea_c": e_i}


class ElectrochemModel:
    name = "electrochemical"
    needs_activity = True

    def __init__(self, p: ParamSet, mult: float, mass_transfer: bool = False) -> None:
        self.p, self.mult, self.mt = p, mult, mass_transfer
        self.use_act = p["ec_use_activities"] > 0.5
        a_ref = 1.0
        if self.use_act:
            ln_oh = pitzer.ln_gamma(1.0, 0.0, T_REF)[1]
            a_ref = math.exp(ln_oh) * 1.0
        explicit = all(k in p.v for k in ("ec_i0a", "ec_i0c", "ec_Ea_a", "ec_Ea_c", "ec_gamma_c"))
        d = {k: p[k] for k in ("ec_i0a", "ec_i0c", "ec_Ea_a", "ec_Ea_c", "ec_gamma_c")}
        if not explicit:
            d.update(from_arrhenius(p["k25"], p["Ea"], p["n_oh"], p, a_ref))
        self.d = d
        self.par = CouplePar(i0a=d["ec_i0a"], i0c=d["ec_i0c"], alpha_a=p["ec_alpha_a"],
                             beta_a=p["ec_alpha_a_back"], alpha_c=p["ec_alpha_c"],
                             beta_c=p["ec_alpha_c_back"], i0c_noble=p["ec_i0c_noble"],
                             galv_ratio=p["ec_galv_ratio"], i_passive=p["ec_i_passive"])
        self.ctx: tuple[float, float] | None = None  # (a_oh, a_al) provided by the engine
        self.n = p["n_oh"]
        self._last: dict[str, float] = {}

    # --- state ------------------------------------------------------------------------------
    def _i0(self, T: float, a_oh: float) -> tuple[float, float, float]:
        p, d = self.p, self.d
        x = 1.0 / T - 1.0 / T_REF
        i0a = d["ec_i0a"] * math.exp(-d["ec_Ea_a"] / R * x) * a_oh ** p["ec_gamma_a"]
        i0c = d["ec_i0c"] * math.exp(-d["ec_Ea_c"] / R * x) * a_oh ** d["ec_gamma_c"]
        i0n = p["ec_i0c_noble"] * math.exp(-d["ec_Ea_c"] / R * x) * a_oh ** d["ec_gamma_c"]
        return i0a, i0c, i0n

    def couple(self, T: float, a_oh: float, a_al: float) -> tuple[float, float]:
        """(E_corr [V], i_corr [A m-2]) for the bare surface."""
        p = self.p
        a_oh = max(a_oh, 1.0e-9)  # magic: floor
        a_al = max(a_al, p["ec_aAl_floor"])
        e0a = p["ec_E0_Al"] + p["ec_dE0_dT_Al"] * (T - T_REF)
        e0c = p["ec_E0_HER"] + p["ec_dE0_dT_HER"] * (T - T_REF)
        e_a, e_c = nernst_potentials(T, a_oh, a_al, e0a, e0c)
        i0a, i0c, i0n = self._i0(T, a_oh)
        e, i = solve_corrosion(T, self.par, e_a, e_c, i0a, i0c, i0n)
        self._last = {"E_corr": e, "i_corr": i, "E_a": e_a, "E_c": e_c}
        return e, i

    def last(self) -> dict[str, float]:
        return dict(self._last)

    # --- RateModel interface -----------------------------------------------------------------
    def _acts(self, c: float) -> tuple[float, float]:
        if self.ctx is not None and self.use_act:
            return self.ctx
        return max(c, 0.0), (self.ctx[1] if self.ctx is not None else 0.0)

    def flux(self, T: float, c: float, film: float, theta_b: float, kmt: np.ndarray | float
             ) -> tuple[np.ndarray | float, np.ndarray | float]:
        a_oh, a_al = self._acts(c)
        if c <= 0.0:
            return 0.0, 0.0
        _, i = self.couple(T, a_oh, a_al)
        j = self.mult * film * (1.0 - theta_b) * i / (3.0 * F)
        return j, c

    def film_rate(self, film: float, T: float, c: float) -> float:
        p = self.p
        a_oh = max(self._acts(c)[0], 0.0)
        kd = (p["ec_k_diss"] * a_oh ** p["ec_k_diss_order"]
              * math.exp(-p["ec_Ea_diss"] / R * (1.0 / T - 1.0 / T_REF)))
        return kd * (1.0 - film) - p["ec_k_pass"] * film

    def film_initial(self) -> float:
        return 0.0

    def kr(self, T: float) -> float:
        """Effective surface rate coefficient at c = 1 mol/L (for Damkoehler reporting)."""
        _, i = self.couple(T, 1.0, 0.0)
        return self.mult * i / (3.0 * F)

    def diagnostics(self, T: float, c: float, c_al: float = 0.0) -> dict[str, float]:
        a_oh, a_al = self._acts(c)
        e, i = self.couple(T, a_oh, a_al)
        return {"E_corr": e, "i_corr": i}


def effective_arrhenius(model: ElectrochemModel, temps: Any = None, concs: Any = None) -> dict[str, float]:
    """Least-squares fit of ln j = ln k25 - Ea/R (1/T-1/T25) + n ln c to the mixed-potential model
    (ideal activities, film fully active) over a T/c grid - the 'limiting-case' Arrhenius parameters."""
    temps = np.linspace(288.15, 348.15, 7) if temps is None else np.asarray(temps)
    concs = np.array([0.5, 1.0, 2.0, 3.0, 4.0]) if concs is None else np.asarray(concs)
    rows, y = [], []
    for T in temps:
        for c in concs:
            model.ctx = (float(c), 0.0)
            j, _ = model.flux(float(T), float(c), 1.0, 0.0, 0.0)
            rows.append([1.0, -(1.0 / T - 1.0 / T_REF) / (1.0 / R), math.log(c)])
            rows[-1][1] = -(1.0 / T - 1.0 / T_REF)
            y.append(math.log(float(j)))
    sol, *_ = np.linalg.lstsq(np.array(rows), np.array(y), rcond=None)
    pred = np.array(rows) @ sol
    return {"k25": float(math.exp(sol[0])), "Ea": float(sol[1] * R), "n": float(sol[2]),
            "max_rel_err": float(np.max(np.abs(np.exp(pred - np.array(y)) - 1.0)))}


def polarization_curve(model: ElectrochemModel, T: float, a_oh: float, a_al: float | None = None,
                       n: int = 200) -> dict[str, np.ndarray | float]:
    """Partial anodic/cathodic current densities versus potential (for Evans diagrams)."""
    from .mixed_potential import _terms

    p = model.p
    a_al_e = max(a_al if a_al is not None else 0.0, p["ec_aAl_floor"])
    e_a, e_c = nernst_potentials(T, a_oh, a_al_e, p["ec_E0_Al"], p["ec_E0_HER"])
    i0a, i0c, i0n = model._i0(T, a_oh)
    es = np.linspace(e_a - 0.1, e_c + 0.1, n)
    ia = np.empty(n)
    ic = np.empty(n)
    for k, e in enumerate(es):
        a, _, c, _ = _terms(float(e), T, model.par, e_a, e_c, i0a, i0c, i0n)
        ia[k], ic[k] = a, c
    ecorr, icorr = solve_corrosion(T, model.par, e_a, e_c, i0a, i0c, i0n)
    return {"E": es, "i_anodic": ia, "i_cathodic": -ic, "E_corr": ecorr, "i_corr": icorr, "E_a": e_a, "E_c": e_c}


def tafel_limit(model: ElectrochemModel, T: float, a_oh: float, a_al: float) -> float:
    """Analytic Tafel-limit corrosion current for comparison with the numerical BV solution."""
    p = model.p
    e_a, e_c = nernst_potentials(T, a_oh, max(a_al, p["ec_aAl_floor"]), p["ec_E0_Al"], p["ec_E0_HER"])
    i0a, i0c, _ = model._i0(T, a_oh)
    return tafel_icorr(T, model.par, e_a, e_c, i0a, i0c)
