"""Joint calibration: least squares -> Fisher information -> profile likelihood (reduced model, fast)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from ..core import l1_fast as lf
from ..core.params import ParamSet
from .data import Channel, Experiment, ParamSpace


class Predictor:
    """Maps parameters to predicted channel readings for a list of experiments (reduced model)."""

    def __init__(self, experiments: list[Experiment], base: ParamSet | None = None, model: str = "empirical",
                 dt: float = 2.0, backend: str = "numpy") -> None:
        self.exps = experiments
        self.base = base or ParamSet()
        self.model, self.dt, self.backend = model, dt, backend

    def sim(self, exp: Experiment, params: ParamSet) -> dict[str, np.ndarray]:
        return lf.simulate_fast(lf.build_constants(exp.scenario, params, self.model, self.dt), exp.scenario, self.dt, self.backend)

    @staticmethod
    def channel_prediction(ch: Channel, sim: dict[str, np.ndarray]) -> np.ndarray:
        if ch.kind == "n_h2":
            return np.interp(ch.t, sim["t"], sim["gen"])
        key = "T" if ch.kind == "T" else "Tw"
        if ch.sensor is not None:
            read = ch.sensor.apply(sim["t"], sim[key])
            return np.interp(ch.t, sim["t"], read)
        return np.interp(ch.t, sim["t"], sim[key])

    def predict(self, params: ParamSet) -> list[list[np.ndarray]]:
        out = []
        for e in self.exps:
            s = self.sim(e, params)
            out.append([self.channel_prediction(c, s) for c in e.channels])
        return out

    def residuals(self, params: ParamSet, model_err_rel: float = 0.0, noise_scale: float = 1.0) -> np.ndarray:
        """Normalised residuals (obs - pred)/sigma_total over all experiments and channels."""
        res = []
        for e, preds in zip(self.exps, self.predict(params), strict=True):
            for c, pr in zip(e.channels, preds, strict=True):
                s = noise_scale * np.sqrt(c.sigma**2 + (model_err_rel * pr) ** 2)
                res.append(np.sqrt(e.weight) * (c.y - pr) / s)
        return np.concatenate(res)

    def residuals_and_logsigma(self, params: ParamSet, model_err_rel: float = 0.0, noise_scale: float = 1.0
                               ) -> tuple[np.ndarray, float]:
        """Normalised residuals and the sum of log total sigmas from ONE set of simulations."""
        res, logs = [], 0.0
        for e, preds in zip(self.exps, self.predict(params), strict=True):
            for c, pr in zip(e.channels, preds, strict=True):
                s = noise_scale * np.sqrt(c.sigma**2 + (model_err_rel * pr) ** 2)
                res.append(np.sqrt(e.weight) * (c.y - pr) / s)
                logs += float(np.sum(np.log(s)))
        return np.concatenate(res), logs

    def log_sigma_total(self, params: ParamSet, model_err_rel: float = 0.0, noise_scale: float = 1.0) -> float:
        return self.residuals_and_logsigma(params, model_err_rel, noise_scale)[1]


@dataclass
class FitResult:
    names: list[str]
    theta: dict[str, float]
    cov: np.ndarray  # covariance in the unconstrained (u) space
    chi2: float
    dof: int
    n: int
    aic: float
    bic: float
    jac: np.ndarray
    residuals: np.ndarray
    success: bool
    space: ParamSpace
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def red_chi2(self) -> float:
        return self.chi2 / max(self.dof, 1)

    def sd(self) -> dict[str, float]:
        """Approximate 1-sigma on the raw parameter scale (delta method)."""
        a = self.space.jac_natural()
        sd_u = np.sqrt(np.diag(a @ self.cov @ a.T))
        scale = np.where(self.space.is_log, np.array([self.theta[n] for n in self.names]), 1.0)
        return {n: float(s * c) for n, s, c in zip(self.names, sd_u, scale, strict=True)}


def fit_lsq(pred: Predictor, names: list[str], x0: dict[str, float] | None = None, model_err_rel: float = 0.0,
            max_nfev: int = 200) -> FitResult:
    """Bounded non-linear least squares over the named parameters (log-transformed where prior is log-normal)."""
    space = ParamSpace.for_experiments(names, pred.exps)
    u0 = space.to_u(x0) if x0 else space.default_u()

    def fun(u: np.ndarray) -> np.ndarray:
        return pred.residuals(pred.base.with_(**space.from_u(u)), model_err_rel)

    sol = least_squares(fun, u0, bounds=(space.lo, space.hi), x_scale="jac", max_nfev=max_nfev, xtol=1e-14, ftol=1e-14, gtol=1e-14)
    j = sol.jac
    n = len(sol.fun)
    chi2 = float(np.sum(sol.fun**2))
    try:
        cov = np.linalg.inv(j.T @ j)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(j.T @ j)
    return FitResult(names, space.from_u(sol.x), cov, chi2, n - len(names), n, chi2 + 2 * len(names),
                     chi2 + len(names) * np.log(n), j, sol.fun, bool(sol.success), space,
                     {"nfev": sol.nfev, "message": sol.message})


# ---------------------------------------------------------------- identifiability
def fisher_information(fit: FitResult) -> dict[str, Any]:
    """FIM = J^T J (unit-sigma residuals), eigen-spectrum, correlations and per-parameter CV."""
    fim_x = fit.jac.T @ fit.jac  # coordinates are O(1) and decorrelated: condition number is meaningful
    w, v = np.linalg.eigh(fim_x)
    a = fit.space.jac_natural()
    cov = a @ np.linalg.pinv(fim_x) @ a.T  # covariance of the natural (log/raw) parameters
    fim = np.linalg.pinv(cov)
    sd = np.sqrt(np.maximum(np.diag(cov), 0.0))
    corr = cov / np.outer(sd, sd + 1e-300)  # magic: floor
    cond = float(w.max() / max(w.min(), 1e-300))  # magic: floor
    cv = {}
    for i, nme in enumerate(fit.names):
        scale = 1.0 if fit.space.is_log[i] else max(abs(fit.theta[nme]), 1e-300)  # magic: floor
        cv[nme] = float(sd[i] / scale)  # relative error (log params: sd of ln)
    weak = [fit.names[int(np.argmax(np.abs(v[:, 0])))]] if w[0] < 1e-6 * w[-1] else []  # magic: weak direction
    return {"fim": fim, "eigvals": w, "eigvecs": v, "corr": corr, "cond": cond, "cv": cv,
            "unidentifiable": [n for n, c in cv.items() if c > 1.0] + weak}


def profile_likelihood(pred: Predictor, fit: FitResult, name: str, n_grid: int = 15, span: float = 3.0,
                       model_err_rel: float = 0.0) -> dict[str, np.ndarray | float]:
    """Profile chi2 of one parameter (others re-optimised); 95 % interval from delta-chi2 = 3.84."""
    i = fit.names.index(name)
    from .data import ParamSpace

    space = ParamSpace(fit.names)  # natural coordinates: the profiled parameter is fixed exactly
    a = fit.space.jac_natural()
    sd_nat = np.sqrt(np.diag(a @ fit.cov @ a.T)) / space.scale
    u_hat = space.to_u(fit.theta)
    sd_u = float(sd_nat[i])
    grid = np.linspace(u_hat[i] - span * sd_u, u_hat[i] + span * sd_u, n_grid)
    grid = np.clip(grid, space.lo[i], space.hi[i])
    others = [k for k in range(len(fit.names)) if k != i]
    chi = np.zeros(n_grid)
    for g, val in enumerate(grid):
        def fun(w: np.ndarray, val: float = val) -> np.ndarray:
            u = u_hat.copy()
            u[others] = w
            u[i] = val
            return pred.residuals(pred.base.with_(**space.from_u(u)), model_err_rel)

        if others:
            s = least_squares(fun, u_hat[others], bounds=(space.lo[others], space.hi[others]), x_scale="jac")
            chi[g] = float(np.sum(s.fun**2))
        else:
            chi[g] = float(np.sum(fun(np.array([])) ** 2))
    d = chi - fit.chi2
    inside = grid[d <= 3.84]  # magic: chi2(1) 95 % quantile
    lo, hi = (float(inside.min()), float(inside.max())) if inside.size else (float("nan"), float("nan"))
    raw = lambda x: space.from_u(np.where(np.arange(len(fit.names)) == i, x, u_hat))[name]  # noqa: E731
    flat = bool(np.ptp(d) < 3.84 and lo <= grid[0] + 1e-12 and hi >= grid[-1] - 1e-12)  # magic: chi2 threshold
    return {"u": grid, "value": np.array([raw(g) for g in grid]), "chi2": chi, "delta": d,
            "ci95": (raw(lo) if inside.size else float("nan"), raw(hi) if inside.size else float("nan")),
            "flat": flat}
