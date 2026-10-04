r"""Kennedy-O'Hagan model discrepancy: ``y = eta(x, theta) + delta(x) + eps`` with a Gaussian-process delta.

Systematic model error is learned explicitly instead of being absorbed into wrong parameter values. The GP
(ARD RBF + noise) lives on the experimental inputs ``(time, T0, [NaOH], size, mass)``; the report shows WHERE
(temperature, molarity, particle size, time window) the physics model fails - a guide to which physics to
improve next.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

from ..core.params import ParamSet
from .calibrate import Predictor
from .data import Experiment, ParamSpace

FEATURES = ("t_s", "T0_C", "c_naoh_M", "dim_um", "al_mass_g")


def _x_of(e: Experiment, t: np.ndarray) -> np.ndarray:
    sc = e.scenario
    base = np.array([sc.T0_C, sc.c_naoh_M, sc.dim_um, sc.al_mass_g])
    return np.column_stack([t, np.tile(base, (len(t), 1))])


@dataclass
class DiscrepancyTable:
    X: np.ndarray
    resid: np.ndarray  # y - eta
    sigma: np.ndarray
    kind: np.ndarray
    exp_id: np.ndarray


def residual_table(pred: Predictor, params: ParamSet, kinds: tuple[str, ...] = ("n_h2",)) -> DiscrepancyTable:
    xs, rs, ss, ks, ids = [], [], [], [], []
    for e, preds in zip(pred.exps, pred.predict(params), strict=True):
        for c, pr in zip(e.channels, preds, strict=True):
            if c.kind not in kinds:
                continue
            xs.append(_x_of(e, c.t))
            rs.append(c.y - pr)
            ss.append(c.sigma)
            ks += [c.kind] * len(c.t)
            ids += [e.id] * len(c.t)
    return DiscrepancyTable(np.vstack(xs), np.concatenate(rs), np.concatenate(ss), np.array(ks), np.array(ids))


class DiscrepancyGP:
    def __init__(self, table: DiscrepancyTable, seed: int = 0) -> None:
        self.table = table
        self.mu = table.X.mean(axis=0)
        self.sd = table.X.std(axis=0) + 1e-12  # magic: floor
        self.scale_y = float(np.std(table.resid) + 1e-12)  # magic: floor
        kern = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(np.ones(table.X.shape[1]), (1e-1, 1e2)) + WhiteKernel(0.1, (1e-6, 1e1))  # magic: hyper-parameter bounds
        self.gp = GaussianProcessRegressor(kern, alpha=(table.sigma / self.scale_y) ** 2, normalize_y=False,
                                           n_restarts_optimizer=2, random_state=seed)
        self.gp.fit((table.X - self.mu) / self.sd, table.resid / self.scale_y)

    def predict(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m, s = self.gp.predict((np.atleast_2d(x) - self.mu) / self.sd, return_std=True)
        return m * self.scale_y, s * self.scale_y

    def nlml(self) -> float:
        return float(-self.gp.log_marginal_likelihood_value_)


def discrepancy_report(table: DiscrepancyTable, gp: DiscrepancyGP, n_bins: int = 4) -> dict[str, Any]:
    """Mean |delta| / sigma in bins of every input: large bins mark where the physics model is deficient."""
    delta, _ = gp.predict(table.X)
    out: dict[str, Any] = {"overall_rms_delta": float(np.sqrt(np.mean(delta**2))),
                           "overall_rms_resid": float(np.sqrt(np.mean(table.resid**2))),
                           "median_sigma": float(np.median(table.sigma)), "by_feature": {}}
    for j, name in enumerate(FEATURES):
        col = table.X[:, j]
        edges = np.unique(np.quantile(col, np.linspace(0, 1, n_bins + 1)))
        rows = []
        for a, b in zip(edges[:-1], edges[1:], strict=True):
            m = (col >= a) & (col <= b) if b == edges[-1] else (col >= a) & (col < b)
            if m.sum() == 0:
                continue
            rows.append({"lo": float(a), "hi": float(b), "n": int(m.sum()),
                         "mean_delta_over_sigma": float(np.mean(np.abs(delta[m]) / table.sigma[m])),
                         "mean_resid_over_sigma": float(np.mean(np.abs(table.resid[m]) / table.sigma[m]))})
        out["by_feature"][name] = rows
    worst = max(((n, r) for n, rr in out["by_feature"].items() for r in rr), key=lambda t: t[1]["mean_delta_over_sigma"])
    out["worst_region"] = {"feature": worst[0], **worst[1]}
    return out


def koh_calibrate(pred: Predictor, names: list[str], x0: dict[str, float] | None = None, iters: int = 3,
                  kinds: tuple[str, ...] = ("n_h2",)) -> dict[str, Any]:
    """Alternating KOH calibration: (theta | GP hyper-parameters) by marginal likelihood, then refit the GP.

    Returns theta, the GP, and the NLML history. The GP absorbs smooth systematic misfit so theta is not
    pushed to compensate for missing physics."""
    space = ParamSpace.for_experiments(names, pred.exps)
    u = space.to_u(x0) if x0 else space.default_u()
    hist = []
    gp: DiscrepancyGP | None = None
    for _ in range(iters):
        theta = pred.base.with_(**space.from_u(u))
        table = residual_table(pred, theta, kinds)
        gp = DiscrepancyGP(table)
        k = gp.gp.kernel_

        def nlml(uu: np.ndarray, k: Any = k, gp: DiscrepancyGP = gp) -> float:
            tab = residual_table(pred, pred.base.with_(**space.from_u(uu)), kinds)
            kmat = k(((tab.X - gp.mu) / gp.sd)) * gp.scale_y**2
            kmat = kmat + np.diag(tab.sigma**2)
            try:
                lfac = np.linalg.cholesky(kmat)
            except np.linalg.LinAlgError:
                return 1e12  # magic: penalty
            z = np.linalg.solve(lfac, tab.resid)
            return float(0.5 * z @ z + np.sum(np.log(np.diag(lfac))))

        sol = minimize(nlml, u, method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 400})
        u = sol.x
        hist.append(float(sol.fun))
    theta_hat = pred.base.with_(**space.from_u(u))
    table = residual_table(pred, theta_hat, kinds)
    gp = DiscrepancyGP(table)
    return {"theta": space.from_u(u), "gp": gp, "table": table, "nlml_history": hist}
