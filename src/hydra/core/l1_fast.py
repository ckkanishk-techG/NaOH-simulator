r"""Independent, fast, differentiable re-implementation of the reduced L1 model.

Single size class, open (atmospheric) vessel, lumped liquid + wall, no boiling/evaporation/valve:

.. math:: \dot f=-R/n_0,\quad \dot n_{AlO}=R,\quad \dot G=\tfrac32 R,\quad
          C(\cdot)\dot T=-\Delta H(T)R-Q_{lw}-Q_{cool},\quad C_w\dot T_w=Q_{lw}-Q_{amb}

with three interchangeable rate laws (``empirical``, ``mass_transfer``, ``echem`` = Arrhenius law with
aluminate (Nernst) dependence and a dissolution/repassivation film).

The same equations are generated from one source template for three back-ends: NumPy+Numba
(pure-Python fallback), and JAX (``lax.scan``; differentiable). Fixed-step RK4 - the second,
independent integrator used to cross-check the adaptive full model and to power Bayesian inference.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..constants import G_ACC, KELVIN_OFFSET, ML, MW_AL, MW_ALOH4, MW_H2O, MW_NA, SIGMA_SB, T_REF, R
from ..thermo import electrolyte, species
from ..thermo import propdb as DB
from .params import ParamSet
from .rates import alloy_multiplier
from .scenario import Scenario

KN = ["k25", "Ea", "n", "tau", "kmt", "kd", "kp", "eps_al", "mult", "a_floor", "w_emp", "w_mt", "w_ec",
      "n0", "a0", "g", "f_s", "V_l", "nNa", "nW0", "cp_w", "cp_na", "cp_oh", "cp_alo", "cp_al", "c_const",
      "dh298", "dcp", "T_ref", "A_in", "t_w", "k_hd", "h_l", "k_liq", "nu_l", "al_l", "beta", "G_acc",
      "A_out", "H_ves", "emis", "sigma", "a_nu", "a_al", "a_pr", "a_k", "T_amb", "C_wall", "UA_cool",
      "T_cool", "stir", "R_gas", "kd_order", "adiabatic"]
KI = {n: i for i, n in enumerate(KN)}
NSTATE = 6  # f, nAlO, T, Tw, psi, gen

_RHS_TEMPLATE = '''
def rhs(y, k):
    f = {MAX}(y[0], 1.0e-12)
    nAlO = y[1]
    T = y[2]
    Tw = y[3]
    psi = y[4]
    nOH = {MAX}(k[{nNa}] - nAlO, 0.0)
    c = nOH / (k[{V_l}] * 1000.0)
    c_al = nAlO / (k[{V_l}] * 1000.0)
    ramp = {MIN}(f / k[{f_s}], 1.0)
    area = k[{a0}] * f ** k[{g}] * ramp
    x = 1.0 / T - 1.0 / k[{T_ref}]
    kr = k[{k25}] * k[{mult}] * {M}.exp(-k[{Ea}] / k[{R_gas}] * x)
    cc = {MAX}(c, 1.0e-12)
    j_emp = kr * psi * cc ** k[{n}]
    # mass-transfer series resistance: kr' cs^n = km (c - cs)
    krp = kr * psi
    km = k[{kmt}] * 1000.0
    cs = c / (1.0 + krp * cc ** (k[{n}] - 1.0) / km)
{NEWTON}
    j_mt = krp * {MAX}(cs, 1.0e-12) ** k[{n}]
    j_ec = j_emp * ({MAX}(c_al, k[{a_floor}]) / k[{a_floor}]) ** (-k[{eps_al}])
    j = k[{w_emp}] * j_emp + k[{w_mt}] * j_mt + k[{w_ec}] * j_ec
    rate = j * area
    df = -rate / k[{n0}]
    dnalo = rate
    dgen = 1.5 * rate
    kd_eff = k[{kd}] * cc ** k[{kd_order}] * {M}.exp(-40000.0 / k[{R_gas}] * x)
    dpsi = (1.0 - k[{w_ec}]) * (1.0 - psi) / k[{tau}] + k[{w_ec}] * (kd_eff * (1.0 - psi) - k[{kp}] * psi)
    # --- energy
    dh = k[{dh298}] + k[{dcp}] * (T - k[{T_ref}])
    q_rx = -dh * rate
    nal_left = f * k[{n0}]
    n_w = k[{nW0}] - 3.0 * (k[{n0}] - nal_left)
    c_tot = (n_w * k[{cp_w}] + k[{nNa}] * k[{cp_na}] + nOH * k[{cp_oh}] + nAlO * k[{cp_alo}]
             + nal_left * k[{cp_al}] + k[{c_const}])
    h_l = k[{h_l}]
    dt_in = {MAX}({ABS}(T - Tw), 0.1)
    ra = k[{G_acc}] * k[{beta}] * dt_in * h_l ** 3 / (k[{nu_l}] * k[{al_l}])
    nu_in = (0.825 + 0.387 * ra ** (1.0 / 6.0) / (1.0 + (0.492 * k[{al_l}] / k[{nu_l}]) ** (9.0 / 16.0)) ** (8.0 / 27.0)) ** 2
    h_in = nu_in * k[{k_liq}] / h_l * k[{stir}]
    r_in = 1.0 / (h_in * k[{A_in}]) + k[{t_w}] / (2.0 * k[{k_hd}] * k[{A_in}])
    q_lw = (T - Tw) / r_in
    q_cool = k[{UA_cool}] * (T - k[{T_cool}])
    t_f = 0.5 * (Tw + k[{T_amb}])
    dt_o = {MAX}({ABS}(Tw - k[{T_amb}]), 0.1)
    ra_o = k[{G_acc}] / t_f * dt_o * k[{H_ves}] ** 3 / (k[{a_nu}] * k[{a_al}])
    nu_o = (0.825 + 0.387 * ra_o ** (1.0 / 6.0) / (1.0 + (0.492 / k[{a_pr}]) ** (9.0 / 16.0)) ** (8.0 / 27.0)) ** 2
    h_o = nu_o * k[{a_k}] / k[{H_ves}]
    q_amb = (Tw - k[{T_amb}]) * h_o * k[{A_out}] / (1.0 + h_o * k[{t_w}] / (2.0 * k[{k_hd}]))
    q_amb = q_amb + k[{emis}] * k[{sigma}] * k[{A_out}] * (Tw ** 4 - k[{T_amb}] ** 4)
    q_amb = q_amb * (1.0 - k[{adiabatic}])
    dT = (q_rx - q_lw - q_cool) / c_tot
    dTw = (q_lw - q_amb) / k[{C_wall}]
    return {ARR}([df, dnalo, dT, dTw, dpsi, dgen])
'''

_NUMPY_INT = '''
def integrate(y0, k, dt, nsteps):
    out = np.empty((nsteps + 1, 6))
    y = y0.copy()
    out[0, :] = y
    for s in range(nsteps):
        k1 = rhs(y, k)
        k2 = rhs(y + 0.5 * dt * k1, k)
        k3 = rhs(y + 0.5 * dt * k2, k)
        k4 = rhs(y + dt * k3, k)
        y = y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        out[s + 1, :] = y
    return out
'''

_JAX_INT = '''
def integrate(y0, k, dt, nsteps):
    def body(y, _):
        k1 = rhs(y, k)
        k2 = rhs(y + 0.5 * dt * k1, k)
        k3 = rhs(y + 0.5 * dt * k2, k)
        k4 = rhs(y + dt * k3, k)
        yn = y + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return yn, yn
    _, ys = lax.scan(body, y0, None, length=nsteps)
    return jnp.concatenate([y0[None, :], ys], axis=0)
'''


_NEWTON_NUMPY = """    for _ in range(8):
        gg = krp * max(cs, 1.0e-12) ** k[{n}] - km * (c - cs)
        dg = krp * k[{n}] * max(cs, 1.0e-12) ** (k[{n}] - 1.0) + km
        cs = min(max(cs - gg / dg, 0.0), c)"""
_NEWTON_JAX = """    def _newton(_i, cs):
        gg = krp * jnp.maximum(cs, 1.0e-12) ** k[{n}] - km * (c - cs)
        dg = krp * k[{n}] * jnp.maximum(cs, 1.0e-12) ** (k[{n}] - 1.0) + km
        return jnp.minimum(jnp.maximum(cs - gg / dg, 0.0), c)
    cs = lax.fori_loop(0, 8, _newton, cs)"""


def _source(backend: str) -> str:
    idx = {n: i for i, n in enumerate(KN)}
    if backend == "jax":
        sub = {"M": "jnp", "MAX": "jnp.maximum", "MIN": "jnp.minimum", "ABS": "jnp.abs", "ARR": "jnp.stack"}
        newton = _NEWTON_JAX
    else:
        sub = {"M": "math", "MAX": "max", "MIN": "min", "ABS": "abs", "ARR": "np.array"}
        newton = _NEWTON_NUMPY
    sub["NEWTON"] = newton.format(**idx)
    return _RHS_TEMPLATE.format(**sub, **idx) + (_JAX_INT if backend == "jax" else _NUMPY_INT)


def _build_numpy() -> tuple[Any, Any]:
    ns: dict[str, Any] = {"np": np, "math": math}
    try:
        import numba

        ns["math"] = math
        exec(_source("numpy"), ns)  # noqa: S102
        rhs = numba.njit(cache=False)(ns["rhs"])
        ns["rhs"] = rhs
        exec(_NUMPY_INT, ns)  # noqa: S102  re-bind integrate to the jitted rhs
        integ = numba.njit(cache=False)(ns["integrate"])
        return rhs, integ
    except Exception:  # pragma: no cover - numba missing/unsupported: pure python fallback
        ns = {"np": np, "math": math}
        exec(_source("numpy"), ns)  # noqa: S102
        return ns["rhs"], ns["integrate"]


_NP_RHS, _NP_INT = _build_numpy()
_JAX_CACHE: dict[str, Any] = {}


def _jax_funcs() -> tuple[Any, Any]:
    if "rhs" not in _JAX_CACHE:
        import jax
        import jax.numpy as jnp
        from jax import lax

        jax.config.update("jax_enable_x64", True)
        ns: dict[str, Any] = {"jnp": jnp, "lax": lax}
        exec(_source("jax"), ns)  # noqa: S102
        _JAX_CACHE["rhs"] = ns["rhs"]
        _JAX_CACHE["int"] = ns["integrate"]
        _JAX_CACHE["int_jit"] = jax.jit(ns["integrate"], static_argnums=(3,))
    return _JAX_CACHE["rhs"], _JAX_CACHE["int_jit"]


def jax_available() -> bool:
    try:
        import jax  # noqa: F401

        return True
    except Exception:
        return False


MODELS = {"empirical": (1.0, 0.0, 0.0), "mass_transfer": (0.0, 1.0, 0.0), "echem": (0.0, 0.0, 1.0)}


def build_constants(sc: Scenario, p: ParamSet | None = None, model: str = "empirical", dt: float = 2.0) -> np.ndarray:
    """Pack scenario + parameters into the constant vector ``k`` used by the fast model.

    The sheet-vanishing ramp width ``f_s`` is chosen so explicit RK4 stays stable (lambda dt < ~2.8 with
    lambda = c/f_s); the full adaptive/implicit model uses the tiny DB value instead."""
    p = p or ParamSet()
    w_emp, w_mt, w_ec = MODELS[model]
    T0 = sc.T0_C + KELVIN_OFFSET
    V_l = sc.v_liq_mL * ML
    nNa = sc.c_naoh_M * V_l / 1.0e-3
    rho = electrolyte.density(sc.c_naoh_M, T0)
    m_w = rho * V_l - nNa * 39.997e-3  # magic: NaOH molar mass
    n_w0 = m_w / MW_H2O
    area_xs = math.pi * p["r_ves"] ** 2
    h_ves = sc.v_vessel_mL * ML / area_xs
    h_l = V_l / area_xs
    r_o = p["r_ves"] + p["t_ves"]
    k_liq = p["k_liq"]
    cp_m = DB.get("cp_liq_mass")
    nu = electrolyte.viscosity(sc.c_naoh_M, T0) / rho
    alp = k_liq / (rho * cp_m)
    from .geometry import make_bins

    b = make_bins(sc.model_copy(update={"psd": None}))
    n_gas = (101325.0 - 3000.0) * (sc.v_vessel_mL - sc.v_liq_mL) * ML / (R * T0)  # magic: ambient air ~ 1 atm, rough vapour
    c_const = n_gas * species.cp("air(g)")
    kr_hot = p["k25"] * alloy_multiplier(sc.alloy, sc.activator_ppm, p) * math.exp(-p["Ea"] / R * (1.0 / 333.15 - 1.0 / T_REF))  # magic: 60 C reference
    c_est = kr_hot * sc.c_naoh_M ** p["n_oh"] * float(b.a0.sum()) / float(b.n0.sum())
    f_s = float(min(max(1.5 * c_est * dt, 0.01), 0.3))  # magic: stability margin and bounds on the ramp width
    vals = {
        "k25": p["k25"], "Ea": p["Ea"], "n": p["n_oh"], "tau": p["tau_ind"], "kmt": p["k_mt"],
        "kd": p["ec_k_diss"], "kp": p["ec_k_pass"], "eps_al": p["ec_eps_al"],
        "mult": alloy_multiplier(sc.alloy, sc.activator_ppm, p), "a_floor": p["ec_aAl_floor"],
        "w_emp": w_emp, "w_mt": w_mt, "w_ec": w_ec,
        "n0": float(b.n0.sum()), "a0": float(b.a0.sum()), "g": b.g, "f_s": f_s,
        "V_l": V_l, "nNa": nNa, "nW0": n_w0,
        "cp_w": species.cp("H2O(l)"), "cp_na": species.cp("Na+(aq)"), "cp_oh": species.cp("OH-(aq)"),
        "cp_alo": species.cp("Al(OH)4-(aq)"), "cp_al": species.cp("Al(s)"), "c_const": c_const,
        "dh298": species.reaction_enthalpy(T_REF), "dcp": species.reaction_delta_cp(), "T_ref": T_REF,
        "A_in": 2.0 * math.pi * p["r_ves"] * h_l + area_xs, "t_w": p["t_ves"], "k_hd": p["k_hdpe"], "h_l": h_l,
        "k_liq": k_liq, "nu_l": nu, "al_l": alp, "beta": p["rho_beta_T"], "G_acc": G_ACC,
        "A_out": 2.0 * math.pi * r_o * h_ves + 2.0 * math.pi * r_o**2, "H_ves": h_ves, "emis": p["emis_hdpe"],
        "sigma": SIGMA_SB, "a_nu": DB.get("air_nu"), "a_al": DB.get("air_alpha"), "a_pr": DB.get("air_Pr"),
        "a_k": DB.get("air_k"), "T_amb": sc.T_amb_C + KELVIN_OFFSET, "C_wall": p["m_ves"] * p["cp_hdpe"],
        "UA_cool": sc.cooling_UA_W_K, "T_cool": sc.coolant_T_C + KELVIN_OFFSET,
        "stir": p["stirred_h_mult"] if sc.stirred else 1.0, "R_gas": R, "kd_order": p["ec_k_diss_order"],
        "adiabatic": 1.0 if sc.adiabatic else 0.0,
    }
    _ = (MW_AL, MW_NA, MW_ALOH4)
    return np.array([vals[n] for n in KN], float)


def initial_state(sc: Scenario) -> np.ndarray:
    T0 = sc.T0_C + KELVIN_OFFSET
    return np.array([1.0, 0.0, T0, T0, 0.0, 0.0])


def fast_applicable(sc: Scenario) -> list[str]:
    """Reasons the reduced model is NOT a faithful stand-in for the full model (empty = applicable)."""
    why = []
    if sc.mode not in ("open",):
        why.append("vessel is not open: pressure/valve physics are not in the reduced model")
    if sc.psd is not None and sc.psd.n_bins > 1:
        why.append("size distribution collapsed to one class")
    if sc.rate_model == "electrochemical":
        why.append("use model='echem' (reduced) instead of the full mixed-potential model")
    return why


def simulate_fast(k: np.ndarray, sc: Scenario, dt: float = 2.0, backend: str = "numpy") -> dict[str, np.ndarray]:
    """Integrate the reduced model; returns time [s] and states on the dt grid (``T``, ``Tw``, ``gen`` ...)."""
    nsteps = int(round(sc.duration_s / dt))
    y0 = initial_state(sc)
    if backend == "jax":
        import jax.numpy as jnp

        _, integ = _jax_funcs()
        out = np.asarray(integ(jnp.asarray(y0), jnp.asarray(k), dt, nsteps))
    else:
        out = _NP_INT(y0, np.ascontiguousarray(k, dtype=np.float64), dt, nsteps)
    t = np.arange(nsteps + 1) * dt
    return {"t": t, "f": out[:, 0], "nAlO": out[:, 1], "T": out[:, 2], "Tw": out[:, 3], "film": out[:, 4],
            "gen": out[:, 5]}


@dataclass
class FastModel:
    """Convenience wrapper: parameters by name -> predictions, for one scenario."""

    sc: Scenario
    base: ParamSet
    model: str = "empirical"
    dt: float = 2.0
    backend: str = "numpy"

    def predict(self, theta: dict[str, float] | None = None) -> dict[str, np.ndarray]:
        p = self.base.with_(**theta) if theta else self.base
        return simulate_fast(build_constants(self.sc, p, self.model), self.sc, self.dt, self.backend)
