r"""Hierarchical Bayesian model: shared physical parameters + batch-level variation.

For every batch ``b`` (Al batch, NaOH lot, sensor set-up) the selected log-normal parameters are shifted:
:math:`\ln\theta_{b,j} = \ln\theta_j + u_{b,j}`, :math:`u_{b,j}\sim N(0,\tau_j^2)`, :math:`\tau_j\sim` half-normal.
Batch noise is therefore not mistaken for physics (it goes into ``u`` / ``tau``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..core.params import ParamSet
from .bayes import Chain, stretch_move
from .calibrate import Predictor
from .data import Experiment, ParamSpace


@dataclass
class HierResult:
    shared: dict[str, np.ndarray]  # samples of shared parameters (natural)
    batch_effect: dict[str, dict[str, np.ndarray]]  # param -> batch -> samples of u_b (ln multiplier)
    tau: dict[str, np.ndarray]
    acceptance: float


class HierPosterior:
    def __init__(self, exps: list[Experiment], names: list[str], batch_params: list[str], base: ParamSet | None = None,
                 model: str = "empirical", dt: float = 2.0, tau_prior_sd: float = 0.5) -> None:
        assert all(b in names for b in batch_params)
        self.exps, self.names, self.bparams = exps, list(names), list(batch_params)
        self.base = base or ParamSet()
        self.pred = Predictor(exps, self.base, model, dt)
        self.space = ParamSpace.for_experiments(names, exps)
        self.batches = sorted({e.batch for e in exps})
        self.nb, self.q, self.p = len(self.batches), len(batch_params), len(names)
        self.tau_sd = tau_prior_sd
        self.dim = self.p + self.nb * self.q + self.q

    def unpack(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = z[: self.p]
        u = z[self.p: self.p + self.nb * self.q].reshape(self.nb, self.q)
        ln_tau = z[self.p + self.nb * self.q:]
        return x, u, ln_tau

    def __call__(self, z: np.ndarray) -> float:
        x, u, ln_tau = self.unpack(z)
        lp = self.space.log_prior(x)
        if not np.isfinite(lp) or np.any(ln_tau < -6.0) or np.any(ln_tau > 1.5):  # magic: tau in [0.0025, 4.5] (ln units)
            return -np.inf
        tau = np.exp(ln_tau)
        lp += float(np.sum(-0.5 * (u / tau) ** 2 - np.log(tau)))  # u ~ N(0, tau^2)
        lp += float(np.sum(-0.5 * (tau / self.tau_sd) ** 2 + ln_tau))  # half-normal tau (Jacobian of ln tau)
        theta = self.space.from_u(x)
        ll = 0.0
        for e in self.exps:
            ib = self.batches.index(e.batch)
            th = dict(theta)
            for j, b in enumerate(self.bparams):
                th[b] = th[b] * np.exp(u[ib, j])
            p = self.base.with_(**th)
            sim = self.pred.sim(e, p)
            for c in e.channels:
                pr = self.pred.channel_prediction(c, sim)
                ll += -0.5 * float(np.sum(((c.y - pr) / c.sigma) ** 2))
        return lp + ll if np.isfinite(ll) else -np.inf

    def start(self, theta: dict[str, float]) -> np.ndarray:
        return np.concatenate([self.space.to_u(theta), np.zeros(self.nb * self.q), np.log(0.1) * np.ones(self.q)])  # magic: tau start 0.1


def sample_hier(post: HierPosterior, x0: np.ndarray, nsteps: int = 400, burn: int = 200, seed: int = 0) -> HierResult:
    rng = np.random.default_rng(seed)
    nw = 2 * post.dim + 2
    p0 = x0[None, :] + 1e-2 * rng.standard_normal((nw, post.dim))
    ch, _, acc = stretch_move(post, p0, nsteps, seed=seed)
    s = ch[burn:].reshape(-1, post.dim)
    shared_x = s[:, : post.p]
    nat = {n: np.empty(len(s)) for n in post.names}
    for i, xx in enumerate(shared_x):
        for n, v in post.space.from_u(xx).items():
            nat[n][i] = v
    u = s[:, post.p: post.p + post.nb * post.q].reshape(len(s), post.nb, post.q)
    be = {b: {bb: u[:, i, j] for i, bb in enumerate(post.batches)} for j, b in enumerate(post.bparams)}
    tau = {b: np.exp(s[:, post.p + post.nb * post.q + j]) for j, b in enumerate(post.bparams)}
    return HierResult(nat, be, tau, acc)


_ = (Chain, Any)
