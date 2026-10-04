r"""Two-stage PCA + Gaussian-process surrogate with physics constraints, conformal error bounds and a
domain-of-validity checker that falls back to the physics model.

Stage 1 (GP): :math:`\ln t_{1/2}` (time to 50 % conversion) and peak temperature rise from the design features
plus physics-derived scales (isothermal time scale, adiabatic rise, Semenov number). Stage 2 (PCA + GP):
the conversion curve in normalised time :math:`X(t/t_{1/2})` and the temperature-rise curve, given the stage-1
outputs. Constraints enforced on the output: :math:`0\le X\le 1`, monotone, X(0)=0, cumulative H2 = X n_max (mass
conservation by construction), dT(0)=0. Error bounds are conformal (95th percentile of per-case max error on a
held-out calibration split, end to end); coverage on a separate test split is reported.
"""

from __future__ import annotations

import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.neural_network import MLPRegressor

from ..constants import BAR, KELVIN_OFFSET, P_ATM
from ..core import l1_fast as lf
from ..core.l1 import SolverSettings, simulate
from ..core.params import ParamSet
from ..core.scenario import Scenario
from ..provenance import sha256
from .dataset import InputSpace, extended_features

S_MAX = 6.0  # normalised-time horizon of the shape model (t / t_half)  # magic: covers sphere burn-out (~4.9 t_half)
N_S = 81  # magic: points of the normalised-time grid


@dataclass
class SurrogateOutput:
    t: np.ndarray
    h2_mol: np.ndarray
    T_K: np.ndarray
    P_gauge_bar: np.ndarray
    source: str  # 'surrogate' | 'physics'
    in_domain: bool
    reasons: list[str] = field(default_factory=list)
    bounds: dict[str, float] = field(default_factory=dict)
    elapsed_s: float = 0.0


def enforce_constraints(x_curve: np.ndarray) -> np.ndarray:
    """Conversion curve: clip to [0,1], force X(0)=0 and make it monotone non-decreasing."""
    x = np.clip(np.asarray(x_curve, float), 0.0, 1.0)
    x[..., 0] = 0.0
    return np.maximum.accumulate(x, axis=-1)


def x_max_of(feat: np.ndarray) -> np.ndarray:
    """Stoichiometric ceiling of the conversion from the design features: each Al consumes one OH-, so
    ``X <= n_OH / n_Al`` (NaOH-limited charges)."""
    f = np.atleast_2d(feat)
    n_al = np.exp(f[:, 0]) * 1e-3 / 26.9815385e-3  # magic: g -> mol Al
    n_oh = np.exp(f[:, 1]) * np.exp(f[:, 2]) * 1e-3
    return np.minimum(1.0, n_oh / n_al)


def snap_to_stoichiometry(x: np.ndarray, x_max: float, frac: float = 0.97, flat_tol: float = 2.0e-3) -> np.ndarray:
    """If the predicted curve has saturated near the stoichiometric ceiling, remove the regression offset of the
    plateau (Al fully consumed): smooth ramp of ``x_max - x[-1]`` once the curve passes ``frac * x_max``."""
    x = np.minimum(x, x_max)
    if x[-1] >= frac * x_max and abs(x[-1] - x[-3]) < flat_tol:
        lo = frac * x_max
        w = np.clip((x - lo) / max(x[-1] - lo, 1e-12), 0.0, 1.0)  # magic: floor
        x = np.minimum(x + (x_max - x[-1]) * w * w * (3 - 2 * w), x_max)
    return np.maximum.accumulate(x)


def _gp(dim: int, seed: int = 0) -> GaussianProcessRegressor:
    kern = ConstantKernel(1.0, (1e-3, 1e3)) * Matern(np.ones(dim), (1e-2, 1e3), nu=2.5) + WhiteKernel(1e-4, (1e-9, 1e-1))  # magic: hyper-parameter bounds
    return GaussianProcessRegressor(kern, normalize_y=True, n_restarts_optimizer=0, random_state=seed)


class Surrogate:
    def __init__(self, space: InputSpace | None = None, kind: str = "gp", n_comp_x: int = 5, n_comp_t: int = 8,
                 params: ParamSet | None = None, with_pressure: bool = False) -> None:
        self.space = space or InputSpace()
        self.kind, self.n_comp_x, self.n_comp_t, self.with_pressure = kind, n_comp_x, n_comp_t, with_pressure
        self.params = params or ParamSet()
        self.lo, self.hi = self.space.lo_hi()
        self.s_grid = np.linspace(0.0, S_MAX, N_S)
        self.flo = self.fhi = np.zeros(1)
        self.stage1: list[Any] = []
        self.s2: dict[str, list[Any]] = {}
        self.pca: dict[str, PCA] = {}
        self.bounds: dict[str, float] = {}
        self.std_threshold = np.inf
        self.nn_threshold = np.inf
        self.meta: dict[str, Any] = {}
        self._utrain = np.zeros((0, 1))
        self.s1lo = self.s1hi = np.zeros(2)

    # ------------------------------------------------------------------ helpers
    def _u(self, f: np.ndarray) -> np.ndarray:
        return (np.atleast_2d(f) - self.flo) / (self.fhi - self.flo + 1e-12)  # magic: floor

    def _u2(self, u: np.ndarray, s1: np.ndarray) -> np.ndarray:
        return np.column_stack([u, (np.atleast_2d(s1) - self.s1lo) / (self.s1hi - self.s1lo + 1e-12)])  # magic: floor

    def _make(self, dim: int, seed: int) -> Any:
        if self.kind == "gp":
            return _gp(dim, seed)
        return MLPRegressor(hidden_layer_sizes=(64, 64), activation="tanh", max_iter=3000, random_state=seed)

    def _shape_target(self, data: dict[str, Any], idx: np.ndarray) -> np.ndarray:
        t = self.space.t_grid
        return np.array([np.interp(self.s_grid * data["t_half"][i], t, data["X"][i]) for i in idx])

    # ------------------------------------------------------------------ training
    def fit(self, data: dict[str, Any], cal_frac: float = 0.15, test_frac: float = 0.15, seed: int = 0) -> dict[str, Any]:
        rng = np.random.default_rng(seed)
        n = len(data["x"])
        idx = rng.permutation(n)
        n_te, n_ca = int(test_frac * n), int(cal_frac * n)
        te, ca, tr = idx[:n_te], idx[n_te: n_te + n_ca], idx[n_te + n_ca:]
        self.flo, self.fhi = data["feat"].min(axis=0), data["feat"].max(axis=0)
        u = self._u(data["feat"])
        self._utrain = u[tr]
        s1 = np.column_stack([np.log(data["t_half"]), data["dT"].max(axis=1)])
        self.s1lo, self.s1hi = s1[tr].min(axis=0), s1[tr].max(axis=0)
        self.stage1 = [self._make(u.shape[1], k).fit(u[tr], s1[tr][:, k]) for k in range(2)]
        u2 = self._u2(u, s1)
        shapes = self._shape_target(data, np.arange(n))
        self.pca["X"] = PCA(self.n_comp_x).fit(shapes[tr])
        self.s2["X"] = [self._make(u2.shape[1], k).fit(u2[tr], self.pca["X"].transform(shapes[tr])[:, k]) for k in range(self.n_comp_x)]
        self.pca["dT"] = PCA(self.n_comp_t).fit(data["dT"][tr])
        self.s2["dT"] = [self._make(u2.shape[1], k).fit(u2[tr], self.pca["dT"].transform(data["dT"][tr])[:, k]) for k in range(self.n_comp_t)]
        if self.with_pressure:
            self.pca["Pg"] = PCA(self.n_comp_t).fit(data["Pg"][tr])
            self.s2["Pg"] = [self._make(u2.shape[1], k).fit(u2[tr], self.pca["Pg"].transform(data["Pg"][tr])[:, k]) for k in range(self.n_comp_t)]
        errs = self._case_errors(data, ca)
        self.bounds = {k: float(np.quantile(v, 0.95)) for k, v in errs.items()}
        stds = np.array([self._pred_std(ui) for ui in u[ca]])
        self.std_threshold = float(np.quantile(stds, 0.95)) * 1.5  # magic: margin over the calibration 95th percentile
        self.nn_threshold = float(np.quantile(self._nn_dist(u[tr], leave_out=True), 0.99)) * 1.5  # magic: margin
        te_err = self._case_errors(data, te)
        pred_te = self._curves(data["feat"][te])
        self.meta = {"n_train": len(tr), "n_cal": len(ca), "n_test": len(te), "fidelity": data.get("fidelity"),
                     "coverage_test": {k: float(np.mean(v <= self.bounds[k])) for k, v in te_err.items()},
                     "test_rmse_X": float(np.sqrt(np.mean((pred_te["X"] - data["X"][te]) ** 2))),
                     "param_hash": sha256(self.params.v)[:12], "hash": sha256({"b": self.bounds, "n": n})[:16]}
        return self.meta

    # ------------------------------------------------------------------ prediction
    def _stage1(self, u: np.ndarray) -> np.ndarray:
        return np.column_stack([m.predict(u) for m in self.stage1])

    def _curves(self, feat: np.ndarray) -> dict[str, np.ndarray]:
        u = self._u(feat)
        s1 = self._stage1(u)
        u2 = self._u2(u, s1)
        t = self.space.t_grid
        shape = enforce_constraints(self.pca["X"].inverse_transform(np.column_stack([m.predict(u2) for m in self.s2["X"]])))
        th = np.exp(s1[:, 0])
        xm = x_max_of(feat)
        x = np.array([snap_to_stoichiometry(enforce_constraints(np.interp(t / th[i], self.s_grid, shape[i])), float(xm[i]))
                      for i in range(len(th))])
        out = {"X": x, "dT": self.pca["dT"].inverse_transform(np.column_stack([m.predict(u2) for m in self.s2["dT"]])),
               "t_half": th, "dT_peak": s1[:, 1]}
        out["dT"][..., 0] = 0.0
        if self.with_pressure:
            out["Pg"] = self.pca["Pg"].inverse_transform(np.column_stack([m.predict(u2) for m in self.s2["Pg"]]))
        return out

    def _pred_std(self, u1: np.ndarray) -> float:
        if self.kind != "gp":
            return 0.0
        s = [m.predict(u1[None, :], return_std=True)[1][0] for m in self.stage1]
        return float(np.sqrt(np.sum(np.square(s))))

    def _nn_dist(self, u: np.ndarray, leave_out: bool = False) -> np.ndarray:
        d = np.sqrt(((u[:, None, :] - self._utrain[None, :, :]) ** 2).sum(axis=2))
        if leave_out:
            d[d == 0.0] = np.inf
        return d.min(axis=1)

    def _case_errors(self, data: dict[str, Any], idx: np.ndarray) -> dict[str, np.ndarray]:
        pred = self._curves(data["feat"][idx])
        out = {"X": np.max(np.abs(pred["X"] - data["X"][idx]), axis=1), "dT": np.max(np.abs(pred["dT"] - data["dT"][idx]), axis=1)}
        if self.with_pressure:
            out["Pg"] = np.max(np.abs(pred["Pg"] - data["Pg"][idx]), axis=1)
        return out

    def domain_check(self, sc: Scenario) -> tuple[bool, list[str]]:
        """Domain of validity: configuration, input box, nearest-neighbour distance, GP uncertainty, predicted regime."""
        why: list[str] = []
        sp = self.space
        if sc.mode != sp.mode or abs(sc.duration_s - sp.duration_s) > 1e-9 or sc.form not in sp.forms:  # magic: tolerance
            why.append("scenario mode/duration/form differ from the training configuration")
        from .dataset import features_of

        x = features_of(sc)
        for i, n in enumerate(("al_mass", "c_naoh", "v_liq", "T0", "T_amb", "dim", "form")):
            if x[i] < self.lo[i] - 1e-9 or x[i] > self.hi[i] + 1e-9:  # magic: tolerance
                why.append(f"{n} outside training box")
        if why:
            return False, why
        feat = extended_features(sc, self.params)
        u = self._u(feat)[0]
        d = float(self._nn_dist(u[None, :])[0])
        if d > self.nn_threshold:
            why.append(f"far from training data (nn distance {d:.3f} > {self.nn_threshold:.3f})")
        if self.kind == "gp" and not why:
            s = self._pred_std(u)
            if s > self.std_threshold:
                why.append(f"predictive uncertainty {s:.3g} > {self.std_threshold:.3g}")
        if x_max_of(feat)[0] < 0.6:  # magic: below this the 50 % reference of the shape model does not exist
            why.append("NaOH-limited charge (X_max < 0.6): outside the surrogate's training regime")
        if not why:
            c = self._curves(feat[None, :])
            if c["t_half"][0] > sp.duration_s:
                why.append("predicted t_half exceeds the horizon")
            if c["dT_peak"][0] + sc.T0_C > sp.peak_T_max_C:
                why.append("predicted peak temperature beyond the surrogate's valid regime (boiling)")
        return (not why), why

    def predict(self, sc: Scenario, physics_fallback: bool = True, params: ParamSet | None = None) -> SurrogateOutput:
        t0 = time.perf_counter()
        ok, why = self.domain_check(sc)
        t = self.space.t_grid
        if ok or not physics_fallback:
            c = self._curves(extended_features(sc, self.params)[None, :])
            nmax = 1.5 * sc.al_mass_g * 1e-3 / 26.9815385e-3  # magic: g/mol Al
            pg = c["Pg"][0] if "Pg" in c else np.zeros_like(t)
            return SurrogateOutput(t, c["X"][0] * nmax, c["dT"][0] + sc.T0_C + KELVIN_OFFSET, pg, "surrogate", ok, why, dict(self.bounds),
                                   time.perf_counter() - t0)
        return physics_run(sc, t, params or self.params, why, time.perf_counter() - t0)

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @staticmethod
    def load(path: str | Path) -> Surrogate:
        with open(path, "rb") as fh:
            return pickle.load(fh)  # noqa: S301  (local trusted file written by save())


def physics_run(sc: Scenario, t: np.ndarray, params: ParamSet | None, why: list[str], elapsed: float = 0.0) -> SurrogateOutput:
    """Fallback to physics: the reduced model where it is faithful (open vessel), else the full adaptive model."""
    p = params or ParamSet()
    if not lf.fast_applicable(sc) and sc.mode == "open":
        s = lf.simulate_fast(lf.build_constants(sc, p, dt=4.0), sc, dt=4.0)  # magic: step
        return SurrogateOutput(t, np.interp(t, s["t"], s["gen"]), np.interp(t, s["t"], s["T"]), np.zeros_like(t), "physics", False, why, {}, elapsed)
    r = simulate(sc, p, SolverSettings(dt_out=float(t[1] - t[0]), rtol=1e-6))
    return SurrogateOutput(t, np.interp(t, r.t, r["h2_gen_mol"]), np.interp(t, r.t, r["T"]),
                           np.interp(t, r.t, (r["P"] - P_ATM) / BAR), "physics", False, why, {}, elapsed)


# ---------------------------------------------------------------- physics-informed neural ODE (optional, JAX)
class NeuralODE:
    r"""Neural-ODE variant: ``dX/dt = softplus(f_X)(1-X)`` (so 0 <= X <= 1 and monotone BY CONSTRUCTION) and
    ``dT/dt = f_T`` with an MLP vector field of (X, dT, inputs); trained on the same curves by Adam (own
    implementation, no extra dependency)."""

    def __init__(self, space: InputSpace, hidden: int = 32, seed: int = 0) -> None:
        import jax
        import jax.numpy as jnp

        self.space, self.jax, self.jnp = space, jax, jnp
        self.lo, self.hi = space.lo_hi()
        k = jax.random.split(jax.random.PRNGKey(seed), 3)
        d_in = 2 + len(self.lo)

        def init(a: int, b: int, kk: Any) -> Any:
            return jax.random.normal(kk, (a, b)) * np.sqrt(1.0 / a)

        self.params = {"w1": init(d_in, hidden, k[0]), "b1": jnp.zeros(hidden), "w2": init(hidden, hidden, k[1]), "b2": jnp.zeros(hidden),
                       "w3": init(hidden, 2, k[2]) * 0.1, "b3": jnp.array([-2.0, 0.0])}  # magic: initial slow growth

    def _field(self, p: dict[str, Any], state: Any, u: Any) -> Any:
        jnp = self.jnp
        h = jnp.tanh(jnp.concatenate([state, u]) @ p["w1"] + p["b1"])
        h = jnp.tanh(h @ p["w2"] + p["b2"])
        o = h @ p["w3"] + p["b3"]
        return jnp.stack([self.jax.nn.softplus(o[0]) * (1.0 - state[0]), o[1]]) * 5.0  # magic: dimensionless-time scale

    def _rollout(self, p: dict[str, Any], u: Any, n: int) -> Any:
        jnp = self.jnp
        h = 1.0 / (n - 1)

        def step(s: Any, _: Any) -> tuple[Any, Any]:
            k1 = self._field(p, s, u)
            k2 = self._field(p, s + 0.5 * h * k1, u)
            k3 = self._field(p, s + 0.5 * h * k2, u)
            k4 = self._field(p, s + h * k3, u)
            sn = s + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
            return sn, sn

        _, traj = self.jax.lax.scan(step, jnp.zeros(2), None, length=n - 1)
        return jnp.vstack([jnp.zeros((1, 2)), traj])

    def fit(self, data: dict[str, Any], epochs: int = 300, lr: float = 5e-3) -> list[float]:
        jax, jnp = self.jax, self.jnp
        u = jnp.asarray((data["x"] - self.lo) / (self.hi - self.lo))
        xs, dts = jnp.asarray(data["X"]), jnp.asarray(data["dT"] / 20.0)  # magic: temperature scale [K]
        n = xs.shape[1]

        def loss(p: dict[str, Any]) -> Any:
            tr = jax.vmap(lambda uu: self._rollout(p, uu, n))(u)
            return jnp.mean((tr[:, :, 0] - xs) ** 2) + 0.2 * jnp.mean((tr[:, :, 1] - dts) ** 2)  # magic: temperature weight

        vg = jax.jit(jax.value_and_grad(loss))
        m = jax.tree_util.tree_map(jnp.zeros_like, self.params)
        v = jax.tree_util.tree_map(jnp.zeros_like, self.params)
        hist = []
        for e in range(1, epochs + 1):
            val, g = vg(self.params)
            hist.append(float(val))
            m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
            v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
            c1, c2 = 1 - 0.9**e, 1 - 0.999**e  # magic: Adam bias corrections
            self.params = jax.tree_util.tree_map(
                lambda p_, m_, v_, c1=c1, c2=c2: p_ - lr * (m_ / c1) / (jnp.sqrt(v_ / c2) + 1e-8), self.params, m, v)  # magic: Adam epsilon
        return hist

    def predict(self, sc: Scenario) -> dict[str, np.ndarray]:
        from .dataset import features_of

        jnp = self.jnp
        u = jnp.asarray((features_of(sc) - self.lo) / (self.hi - self.lo))
        tr = np.asarray(self._rollout(self.params, u, self.space.n_time))
        nmax = 1.5 * sc.al_mass_g * 1e-3 / 26.9815385e-3  # magic: g/mol Al
        return {"t": self.space.t_grid, "h2_mol": tr[:, 0] * nmax, "T_K": tr[:, 1] * 20.0 + sc.T0_C + KELVIN_OFFSET}
