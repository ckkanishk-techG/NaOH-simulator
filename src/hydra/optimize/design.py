r"""Design optimisation: goal-seeking, differential evolution, Bayesian optimisation, JAX-gradient descent,
robust (posterior) optimisation and multi-objective Pareto fronts (NSGA-II).

Goal: "deliver >= X mL/min (STP) of H2 for Y min with peak T < Z C and relief pressure <= P_max at minimum cost".
Decision variables: Al mass/form/size, NaOH molarity, liquid volume (water), cooling, optional stack size and
buffer battery. Designs are screened with the fast reduced model (open vessel, validated against the full model);
``verify_design`` re-evaluates a result with the full relief-valve model.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, Field
from scipy.optimize import differential_evolution, minimize

from ..constants import BAR, KELVIN_OFFSET, MW_NAOH, P_ATM, F, R
from ..core import l1_fast as lf
from ..core.l1 import SolverSettings, simulate
from ..core.params import ParamSet
from ..core.scenario import Scenario
from ..thermo import propdb as DB
from .cost import Prices, total_cost

FORMS = ["foil", "powder", "wire", "can"]
V_STP = 22.414  # L/mol at 0 C, 1 atm (derived from R*T_STP/P_ATM, see constants) # magic: unit factor


class Goal(BaseModel):
    flow_mL_min: float = Field(50.0, gt=0)  # sustained H2 flow, mL/min at STP
    duration_min: float = Field(30.0, gt=0)
    ramp_s: float = Field(120.0, ge=0)  # start-up allowance
    buffer_mL: float = Field(50.0, ge=0)  # H2 held in the headspace/buffer (STP mL)
    peak_T_C: float = 70.0
    p_max_bar_g: float = 1.5  # relief set point / maximum pressure
    min_sf: float = 3.0
    T0_C: float = 25.0
    T_amb_C: float = 25.0
    i_op_A_cm2: float = 0.3  # stack operating current density used for sizing
    cell_area_cm2: float = 25.0
    battery_Wh_min: float = 0.0
    forms: list[str] = Field(default_factory=lambda: ["foil", "powder"])
    include_stack: bool = False
    include_battery: bool = False


@dataclass
class Space:
    """Decision-variable bounds; vector layout: [ln m_Al, form_idx, ln d, c, V_liq, UA, n_cells, battery_Wh]."""

    al_mass_g: tuple[float, float] = (0.2, 20.0)
    dim_um: tuple[float, float] = (10.0, 400.0)
    c_naoh_M: tuple[float, float] = (0.5, 5.0)
    v_liq_mL: tuple[float, float] = (50.0, 400.0)
    cooling_UA: tuple[float, float] = (0.0, 5.0)
    n_cells: tuple[int, int] = (1, 12)
    battery_Wh: tuple[float, float] = (0.0, 30.0)

    def bounds(self, goal: Goal) -> list[tuple[float, float]]:
        b = [(math.log(self.al_mass_g[0]), math.log(self.al_mass_g[1])), (0, len(goal.forms) - 1e-9),
             (math.log(self.dim_um[0]), math.log(self.dim_um[1])), self.c_naoh_M, self.v_liq_mL, self.cooling_UA]
        if goal.include_stack:
            b.append((self.n_cells[0], self.n_cells[1] + 0.999))
        if goal.include_battery:
            b.append(self.battery_Wh)
        return b


def decode(x: np.ndarray, goal: Goal) -> dict[str, Any]:
    d: dict[str, Any] = {"al_mass_g": float(math.exp(x[0])), "form": goal.forms[int(min(x[1], len(goal.forms) - 1))],
                         "dim_um": float(math.exp(x[2])), "c_naoh_M": float(x[3]), "v_liq_mL": float(x[4]),
                         "cooling_UA": float(x[5]), "n_cells": 0, "battery_Wh": 0.0}
    k = 6
    if goal.include_stack:
        d["n_cells"] = int(x[k])
        k += 1
    if goal.include_battery:
        d["battery_Wh"] = float(x[k])
    return d


def to_scenario(d: dict[str, Any], goal: Goal) -> Scenario:
    return Scenario(al_mass_g=d["al_mass_g"], form=d["form"], dim_um=d["dim_um"], c_naoh_M=d["c_naoh_M"],
                    v_liq_mL=d["v_liq_mL"], v_vessel_mL=max(2.5 * d["v_liq_mL"], d["v_liq_mL"] + 150.0),  # magic: headspace sizing rule
                    T0_C=goal.T0_C, T_amb_C=goal.T_amb_C, mode="open", vent_diameter_mm=10.0,
                    cooling_UA_W_K=max(d["cooling_UA"], 0.0), duration_s=goal.duration_min * 60.0 * 1.05)  # magic: simulate 5 % past the requirement


def relief_capacity_mol_s(params: ParamSet, p_set_bar_g: float, T: float = 320.0) -> float:
    """Orifice capacity of the relief valve for H2 at its set pressure [mol/s] (choked/unchoked ideal-gas orifice)."""
    g, cd = DB.get("gas_gamma"), params["valve_Cd"]
    area = params["valve_Cv"] * DB.get("valve_Av_per_Cv")
    p = P_ATM + p_set_bar_g * BAR
    pr = P_ATM / p
    crit = (2.0 / (g + 1.0)) ** (g / (g - 1.0))
    if pr <= crit:
        psi = math.sqrt(g) * (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))
    else:
        psi = math.sqrt(2.0 * g / (g - 1.0) * (pr ** (2.0 / g) - pr ** ((g + 1.0) / g)))
    return cd * area * p * psi / math.sqrt(R * T * 2.016e-3)  # magic: H2 molar mass kg/mol


def evaluate(x: np.ndarray, goal: Goal, prices: Prices | None = None, params: ParamSet | None = None,
             dt: float = 4.0, model: str = "empirical") -> dict[str, Any]:
    """Simulate a design with the fast model; return metrics and normalised constraint violations (>0 = violated)."""
    prices = prices or Prices()
    params = params or ParamSet()
    d = decode(x, goal)
    sc = to_scenario(d, goal)
    sim = lf.simulate_fast(lf.build_constants(sc, params, model, dt), sc, dt)
    t, gen = sim["t"], sim["gen"]
    y = goal.duration_min * 60.0
    f_mol = goal.flow_mL_min / 1e3 / V_STP / 60.0
    buf = goal.buffer_mL / 1e3 / V_STP
    need = f_mol * np.maximum(t - goal.ramp_s, 0.0)
    m = t <= y
    slack = gen[m] + buf - need[m]
    peak_T = float(sim["T"].max() - KELVIN_OFFSET)
    peak_flow = float(np.max(np.gradient(gen, t))) if len(t) > 2 else 0.0
    cap = relief_capacity_mol_s(params, goal.p_max_bar_g)
    sy, r_ves, t_ves = params["sy23"], params["r_ves"], params["t_ves"]
    sf = sy / (goal.p_max_bar_g * BAR * r_ves / t_ves)
    viol = {
        "flow": max(0.0, -float(slack.min())) / max(f_mol * y, 1e-12),  # magic: floor
        "peak_T": max(0.0, peak_T - goal.peak_T_C) / 10.0,  # magic: 10 K scale
        "relief": max(0.0, peak_flow / cap - 1.0),
        "hoop_sf": max(0.0, (goal.min_sf - sf) / goal.min_sf),
        "stack": 0.0, "battery": 0.0,
    }
    if goal.include_stack:
        per_cell = goal.i_op_A_cm2 * goal.cell_area_cm2 / (2.0 * F)  # mol/s per cell
        viol["stack"] = max(0.0, 1.0 - d["n_cells"] * per_cell / f_mol)
    if goal.include_battery:
        viol["battery"] = max(0.0, goal.battery_Wh_min - d["battery_Wh"]) / max(goal.battery_Wh_min, 1.0)
    n_naoh = d["c_naoh_M"] * d["v_liq_mL"] / 1e3
    mass_kg = d["al_mass_g"] / 1e3 + n_naoh * MW_NAOH + d["v_liq_mL"] / 1e3 * 1.0 + params["m_ves"] + d["n_cells"] * 0.15  # magic: ~150 g per cell (placeholder)
    cost = total_cost(sc, prices, d["n_cells"], d["battery_Wh"])
    return {"design": d, "scenario": sc, "cost": cost, "peak_T_C": peak_T, "h2_total_mL": float(gen[-1] * V_STP * 1e3),
            "yield_mL_per_g": float(gen[-1] * V_STP * 1e3 / d["al_mass_g"]), "mass_kg": mass_kg,
            "volume_L": sc.v_vessel_mL / 1e3, "violations": viol, "viol_sum": float(sum(viol.values())),
            "feasible": all(v <= 1e-9 for v in viol.values()), "peak_flow_mol_s": peak_flow}  # magic: tolerance


PENALTY = 1.0e4  # INR per unit normalised violation (static penalty)


def penalized(x: np.ndarray, goal: Goal, prices: Prices, params: ParamSet) -> float:
    r = evaluate(x, goal, prices, params)
    return r["cost"] + PENALTY * r["viol_sum"]


@dataclass
class OptResult:
    x: np.ndarray
    design: dict[str, Any]
    metrics: dict[str, Any]
    history: list[float] = field(default_factory=list)
    n_eval: int = 0
    method: str = ""


def _wrap(res_x: np.ndarray, goal: Goal, prices: Prices, params: ParamSet, hist: list[float], n: int, method: str) -> OptResult:
    m = evaluate(res_x, goal, prices, params)
    return OptResult(res_x, m["design"], m, hist, n, method)


def optimize_de(goal: Goal, space: Space | None = None, prices: Prices | None = None, params: ParamSet | None = None,
                seed: int = 0, maxiter: int = 60, popsize: int = 12) -> OptResult:
    """Differential evolution (global) on the penalised cost."""
    space, prices, params = space or Space(), prices or Prices(), params or ParamSet()
    b = space.bounds(goal)
    integ = [False, True, False, False, False, False] + ([True] if goal.include_stack else []) + ([False] if goal.include_battery else [])
    hist: list[float] = []
    cnt = [0]

    def f(x: np.ndarray) -> float:
        cnt[0] += 1
        return penalized(x, goal, prices, params)

    res = differential_evolution(f, b, seed=seed, maxiter=maxiter, popsize=popsize, tol=1e-6, polish=False,
                                 integrality=np.array(integ), callback=lambda xk, convergence=None: hist.append(f(xk)))
    return _wrap(res.x, goal, prices, params, hist, cnt[0], "differential_evolution")


def optimize_bo(goal: Goal, space: Space | None = None, prices: Prices | None = None, params: ParamSet | None = None,
                n_init: int = 20, n_iter: int = 40, seed: int = 0) -> OptResult:
    """Bayesian optimisation: Matern GP on log(penalised cost) with expected-improvement acquisition."""
    from scipy.stats import norm
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import ConstantKernel, Matern

    space, prices, params = space or Space(), prices or Prices(), params or ParamSet()
    rng = np.random.default_rng(seed)
    b = np.array(space.bounds(goal))
    lo, hi = b[:, 0], b[:, 1]

    def f(x: np.ndarray) -> float:
        return float(np.log(penalized(x, goal, prices, params)))

    X = lo + (hi - lo) * rng.random((n_init, len(lo)))
    y = np.array([f(x) for x in X])
    hist = [float(np.exp(y.min()))]
    for _ in range(n_iter):
        gp = GaussianProcessRegressor(ConstantKernel() * Matern(np.ones(len(lo)), nu=2.5), normalize_y=True,
                                      n_restarts_optimizer=2, random_state=int(rng.integers(1 << 30)))
        gp.fit((X - lo) / (hi - lo), y)
        cand = lo + (hi - lo) * rng.random((4000, len(lo)))  # magic: random-search acquisition
        mu, sd = gp.predict((cand - lo) / (hi - lo), return_std=True)
        best = y.min()
        z = (best - mu) / np.maximum(sd, 1e-9)  # magic: floor
        ei = (best - mu) * norm.cdf(z) + sd * norm.pdf(z)
        xn = cand[int(np.argmax(ei))]
        X, y = np.vstack([X, xn]), np.append(y, f(xn))
        hist.append(float(np.exp(y.min())))
    xb = X[int(np.argmin(y))]
    return _wrap(xb, goal, prices, params, hist, len(y), "bayesian_optimization")


class JaxPenalty:
    """Penalised objective differentiated by JAX through the dynamics (``d J / d k``), chained to the
    continuous design variables by central differences of the (cheap, algebraic) constant map ``x -> k``."""

    def __init__(self, goal: Goal, form: str, prices: Prices | None = None, params: ParamSet | None = None, dt: float = 4.0) -> None:
        import jax
        import jax.numpy as jnp

        lf._jax_funcs()
        self.goal, self.form, self.prices, self.params, self.dt = goal, form, prices or Prices(), params or ParamSet(), dt
        self.jax, self.jnp = jax, jnp
        self.nsteps = int(round(goal.duration_min * 60.0 * 1.05 / dt))
        integ = lf._JAX_CACHE["int"]
        t = np.arange(self.nsteps + 1) * dt
        y_req = goal.duration_min * 60.0
        mask = jnp.asarray(t <= y_req)
        f_mol = goal.flow_mL_min / 1e3 / V_STP / 60.0
        buf = goal.buffer_mL / 1e3 / V_STP
        need = jnp.asarray(f_mol * np.maximum(t - goal.ramp_s, 0.0))

        def viol(k: Any, y0: Any) -> Any:
            o = integ(y0, k, dt, self.nsteps)
            slack = jnp.where(mask, o[:, 5] + buf - need, 1e9)  # magic: masked out
            v_flow = jnp.maximum(0.0, -jnp.min(slack)) / (f_mol * y_req)
            v_T = jnp.maximum(0.0, jnp.max(o[:, 2]) - 273.15 - goal.peak_T_C) / 10.0  # magic: 10 K scale
            return v_flow + v_T

        self._grad = jax.jit(jax.value_and_grad(viol))

    def _scenario(self, xc: np.ndarray) -> Scenario:
        d = {"al_mass_g": float(math.exp(xc[0])), "form": self.form, "dim_um": float(math.exp(xc[1])), "c_naoh_M": float(xc[2]),
             "v_liq_mL": float(xc[3]), "cooling_UA": float(xc[4]), "n_cells": 0, "battery_Wh": 0.0}
        return to_scenario(d, self.goal)

    def k_of(self, xc: np.ndarray) -> np.ndarray:
        return lf.build_constants(self._scenario(xc), self.params, "empirical", self.dt)

    def value_and_grad(self, xc: np.ndarray) -> tuple[float, np.ndarray]:
        jnp = self.jnp
        sc = self._scenario(xc)
        k = self.k_of(xc)
        v, gk = self._grad(jnp.asarray(k), jnp.asarray(lf.initial_state(sc)))
        gk = np.asarray(gk)
        grad_v = np.zeros(len(xc))
        for i in range(len(xc)):
            h = 1e-5 * max(abs(xc[i]), 1.0)  # magic: FD step of the algebraic map
            xp, xm = xc.copy(), xc.copy()
            xp[i] += h
            xm[i] -= h
            grad_v[i] = float(gk @ (self.k_of(xp) - self.k_of(xm)) / (2 * h))
        cost_fn = lambda z: total_cost(self._scenario(z), self.prices)  # noqa: E731
        grad_c = np.array([(cost_fn(xc + np.eye(len(xc))[i] * 1e-4) - cost_fn(xc - np.eye(len(xc))[i] * 1e-4)) / 2e-4 for i in range(len(xc))])  # magic: FD step
        return cost_fn(xc) + PENALTY * float(v), grad_c + PENALTY * grad_v


def optimize_gradient(goal: Goal, form: str, x0: np.ndarray, space: Space | None = None, prices: Prices | None = None,
                      params: ParamSet | None = None, maxiter: int = 60) -> OptResult:
    """Projected L-BFGS-B with JAX gradients over the continuous variables (fixed form, no stack/battery).

    ``x0`` = [ln m_Al, ln d, c, V_liq, UA]."""
    space = space or Space()
    jp = JaxPenalty(goal, form, prices, params)
    b = [(math.log(space.al_mass_g[0]), math.log(space.al_mass_g[1])), (math.log(space.dim_um[0]), math.log(space.dim_um[1])),
         space.c_naoh_M, space.v_liq_mL, space.cooling_UA]
    hist: list[float] = []
    scale: list[float] = []  # objective scaled once by the initial gradient norm so L-BFGS-B's unit first step is sane

    def fg(x: np.ndarray) -> tuple[float, np.ndarray]:
        v, g = jp.value_and_grad(x)
        if not scale:
            scale.append(1.0 / max(float(np.linalg.norm(g)), 1.0))  # magic: floor
        if not (math.isfinite(v) and np.all(np.isfinite(g))):  # unstable trial point: large finite value so the line search backtracks
            return 1e3 * (hist[0] if hist else 1.0) * scale[0], np.zeros_like(x)  # magic: penalty for failed trial
        hist.append(v)
        return v * scale[0], g * scale[0]

    res = minimize(fg, x0, jac=True, method="L-BFGS-B", bounds=b, options={"maxiter": maxiter})
    xfull = np.array([res.x[0], goal.forms.index(form), res.x[1], res.x[2], res.x[3], res.x[4]])
    return _wrap(xfull, goal, jp.prices, jp.params, hist, len(hist), "jax_gradient")


# ---------------------------------------------------------------- robust optimisation across the posterior
def robust_penalized(x: np.ndarray, goal: Goal, prices: Prices, draws: list[ParamSet], alpha: float = 0.8) -> float:
    """Cost + penalty on CVaR_alpha of the total violation over posterior parameter draws."""
    v = np.array([evaluate(x, goal, prices, p)["viol_sum"] for p in draws])
    k = max(int(math.ceil((1 - alpha) * len(v))), 1)
    cvar = float(np.sort(v)[-k:].mean())
    cost = evaluate(x, goal, prices, draws[0])["cost"]
    return cost + PENALTY * cvar


def optimize_robust(goal: Goal, draws: list[ParamSet], space: Space | None = None, prices: Prices | None = None,
                    seed: int = 0, maxiter: int = 40, popsize: int = 10, alpha: float = 0.8) -> OptResult:
    space, prices = space or Space(), prices or Prices()
    integ = [False, True, False, False, False, False] + ([True] if goal.include_stack else []) + ([False] if goal.include_battery else [])
    res = differential_evolution(lambda x: robust_penalized(x, goal, prices, draws, alpha), space.bounds(goal), seed=seed,
                                 maxiter=maxiter, popsize=popsize, polish=False, integrality=np.array(integ))
    return _wrap(res.x, goal, prices, draws[0], [], 0, "robust_de")


def feasibility_probability(x: np.ndarray, goal: Goal, prices: Prices, draws: list[ParamSet]) -> float:
    return float(np.mean([evaluate(x, goal, prices, p)["feasible"] for p in draws]))


# ---------------------------------------------------------------- multi-objective (NSGA-II)
OBJECTIVES = {"neg_yield": lambda m: -m["yield_mL_per_g"], "peak_T": lambda m: m["peak_T_C"], "cost": lambda m: m["cost"],
              "mass": lambda m: m["mass_kg"], "volume": lambda m: m["volume_L"]}


def _dominates(a: np.ndarray, b: np.ndarray, va: float, vb: float) -> bool:
    if va == 0 and vb > 0:
        return True
    if va > 0 and vb > 0:
        return va < vb
    if va > 0 and vb == 0:
        return False
    return bool(np.all(a <= b) and np.any(a < b))


def _fronts(F_: np.ndarray, v: np.ndarray) -> list[list[int]]:
    n = len(F_)
    dom: list[list[int]] = [[] for _ in range(n)]
    cnt = np.zeros(n, int)
    for i in range(n):
        for j in range(n):
            if i != j and _dominates(F_[i], F_[j], v[i], v[j]):
                dom[i].append(j)
            elif i != j and _dominates(F_[j], F_[i], v[j], v[i]):
                cnt[i] += 1
    fronts = [[i for i in range(n) if cnt[i] == 0]]
    while fronts[-1]:
        nxt = []
        for i in fronts[-1]:
            for j in dom[i]:
                cnt[j] -= 1
                if cnt[j] == 0:
                    nxt.append(j)
        fronts.append(nxt)
    return fronts[:-1]


def _crowding(F_: np.ndarray) -> np.ndarray:
    n, m = F_.shape
    d = np.zeros(n)
    for k in range(m):
        idx = np.argsort(F_[:, k])
        d[idx[0]] = d[idx[-1]] = np.inf
        rng_ = F_[idx[-1], k] - F_[idx[0], k] or 1.0
        d[idx[1:-1]] += (F_[idx[2:], k] - F_[idx[:-2], k]) / rng_
    return d


def pareto_nsga2(goal: Goal, objectives: tuple[str, ...] = ("neg_yield", "peak_T", "cost"), space: Space | None = None,
                 prices: Prices | None = None, params: ParamSet | None = None, pop: int = 40, gens: int = 25,
                 seed: int = 0) -> list[dict[str, Any]]:
    """NSGA-II with constrained domination. Returns the non-dominated feasible designs with their objectives."""
    space, prices, params = space or Space(), prices or Prices(), params or ParamSet()
    rng = np.random.default_rng(seed)
    b = np.array(space.bounds(goal))
    lo, hi = b[:, 0], b[:, 1]
    discrete = np.zeros(len(lo), bool)
    discrete[1] = True
    if goal.include_stack:
        discrete[6] = True
    cache: dict[bytes, dict[str, Any]] = {}

    def ev(x: np.ndarray) -> dict[str, Any]:
        key = np.round(x, 8).tobytes()
        if key not in cache:
            cache[key] = evaluate(x, goal, prices, params)
        return cache[key]

    def fit(P_: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        ms = [ev(x) for x in P_]
        return np.array([[OBJECTIVES[o](m) for o in objectives] for m in ms]), np.array([m["viol_sum"] for m in ms])

    P = lo + (hi - lo) * rng.random((pop, len(lo)))
    P[:, discrete] = np.floor(P[:, discrete])
    for _ in range(gens):
        Fv, V = fit(P)
        fr = _fronts(Fv, V)
        rank = np.zeros(len(P), int)
        for r, f in enumerate(fr):
            rank[f] = r
        crowd = np.zeros(len(P))
        for f in fr:
            crowd[f] = _crowding(Fv[f])
        children = []
        for _ in range(pop):
            i, j = rng.integers(0, len(P), 2), rng.integers(0, len(P), 2)
            pa = P[i[0]] if (rank[i[0]], -crowd[i[0]]) < (rank[i[1]], -crowd[i[1]]) else P[i[1]]
            pb = P[j[0]] if (rank[j[0]], -crowd[j[0]]) < (rank[j[1]], -crowd[j[1]]) else P[j[1]]
            beta = rng.random(len(lo))
            c = beta * pa + (1 - beta) * pb
            mut = rng.random(len(lo)) < 0.2  # magic: mutation rate
            c = np.where(mut, c + 0.1 * (hi - lo) * rng.standard_normal(len(lo)), c)  # magic: mutation scale
            c = np.clip(c, lo, hi - 1e-9)
            c[discrete] = np.floor(c[discrete])
            children.append(c)
        Q = np.vstack([P, np.array(children)])
        Fq, Vq = fit(Q)
        fr = _fronts(Fq, Vq)
        nxt: list[int] = []
        for f in fr:
            if len(nxt) + len(f) <= pop:
                nxt += f
            else:
                cr = _crowding(Fq[f])
                nxt += [f[k] for k in np.argsort(-cr)[: pop - len(nxt)]]
                break
        P = Q[nxt]
    Fv, V = fit(P)
    front = _fronts(Fv, V)[0]
    out, seen = [], set()
    for idx_f in front:
        m = ev(P[idx_f])
        if not m["feasible"]:
            continue
        key = tuple(np.round(Fv[idx_f], 6))
        if key in seen:
            continue
        seen.add(key)
        out.append({"x": P[idx_f].copy(), "design": m["design"], "objectives": dict(zip(objectives, Fv[idx_f].tolist(), strict=True)),
                    "metrics": m})
    return out


def verify_design(res: OptResult, goal: Goal, params: ParamSet | None = None) -> dict[str, Any]:
    """Re-run the optimised design in the FULL model with the relief valve (and report headline numbers)."""
    sc = res.metrics["scenario"].model_copy(update={"mode": "relief", "p_relief_bar_g": goal.p_max_bar_g, "T0_C": goal.T0_C,
                                                    "v_vessel_mL": res.metrics["scenario"].v_vessel_mL})
    r = simulate(sc, params, SolverSettings(dt_out=5.0, rtol=1e-6))
    return {"peak_T_C": r.summary["peak_T_C"], "peak_P_bar_g": r.summary["peak_P_bar_g"], "h2_stp_L": r.summary["h2_total_stp_L"],
            "min_sf": r.summary["min_sf"], "vent_pct": r.summary["vent_pct"], "result": r}
