r"""Bayesian inference: ensemble MCMC (affine-invariant stretch move; emcee if installed), diagnostics,
credible intervals, posterior-predictive bands/checks, corner data, WAIC.

Posterior: :math:`\log p(x|y) = \log p(x) - \tfrac12\sum_i r_i^2 - \sum_i\ln\sigma_i`  with residuals from
the reduced model (``Predictor``) in decorrelated coordinates ``x`` (see ``ParamSpace``).
Optionally a noise-scale parameter ``ln s`` (flat on [ln 0.25, ln 8]) inflates every sigma.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .calibrate import FitResult, Predictor
from .data import ParamSpace

LN_S_BOUNDS = (float(np.log(0.25)), float(np.log(8.0)))  # magic: prior range of the noise-scale factor


class Posterior:
    def __init__(self, pred: Predictor, names: list[str], model_err_rel: float = 0.0, infer_noise: bool = False) -> None:
        self.pred, self.names = pred, list(names)
        self.space = ParamSpace.for_experiments(names, pred.exps)
        self.model_err_rel, self.infer_noise = model_err_rel, infer_noise
        self.dim = len(names) + (1 if infer_noise else 0)

    def split(self, z: np.ndarray) -> tuple[np.ndarray, float]:
        return (z[:-1], float(np.exp(z[-1]))) if self.infer_noise else (z, 1.0)

    def loglik_terms(self, z: np.ndarray) -> np.ndarray:
        """Pointwise log-likelihood (for WAIC/LOO)."""
        x, s = self.split(z)
        p = self.pred.base.with_(**self.space.from_u(x))
        out = []
        for e, preds in zip(self.pred.exps, self.pred.predict(p), strict=True):
            for c, pr in zip(e.channels, preds, strict=True):
                sig = s * np.sqrt(c.sigma**2 + (self.model_err_rel * pr) ** 2)
                out.append(-0.5 * ((c.y - pr) / sig) ** 2 - np.log(sig) - 0.5 * np.log(2 * np.pi))
        return np.concatenate(out)

    def __call__(self, z: np.ndarray) -> float:
        x, s = self.split(z)
        lp = self.space.log_prior(x)
        if not np.isfinite(lp):
            return -np.inf
        if self.infer_noise and not (LN_S_BOUNDS[0] <= z[-1] <= LN_S_BOUNDS[1]):
            return -np.inf
        p = self.pred.base.with_(**self.space.from_u(x))
        r, logs = self.pred.residuals_and_logsigma(p, self.model_err_rel, s)
        val = lp - 0.5 * float(r @ r) - logs
        return val if np.isfinite(val) else -np.inf

    def start(self, fit: FitResult | None = None) -> np.ndarray:
        x0 = self.space.to_u(fit.theta) if fit is not None else self.space.default_u()
        return np.append(x0, 0.0) if self.infer_noise else x0


# ---------------------------------------------------------------- sampler
@dataclass
class Chain:
    samples: np.ndarray  # (nsteps, nwalkers, dim) after burn-in
    logp: np.ndarray
    acceptance: float
    backend: str
    posterior: Posterior
    diag: dict[str, Any] = field(default_factory=dict)

    def flat(self, thin: int = 1) -> np.ndarray:
        s = self.samples[::thin]
        return s.reshape(-1, s.shape[-1])

    def natural(self, thin: int = 1) -> dict[str, np.ndarray]:
        """Samples of the physical parameters (back-transformed)."""
        fl = self.flat(thin)
        sp = self.posterior.space
        out = {n: np.empty(len(fl)) for n in sp.names}
        for i, z in enumerate(fl):
            x, _ = self.posterior.split(z)
            for n, v in sp.from_u(x).items():
                out[n][i] = v
        return out


def stretch_move(logp: Callable[[np.ndarray], float], x0: np.ndarray, nsteps: int, a: float = 2.0,
                 seed: int = 0) -> tuple[np.ndarray, np.ndarray, float]:
    """Goodman-Weare affine-invariant ensemble sampler (two-half parallel update). Returns (chain, logp, acc)."""
    rng = np.random.default_rng(seed)
    w, d = x0.shape
    assert w % 2 == 0 and w >= 2 * d, "need an even number of walkers >= 2*dim"
    x = x0.copy()
    lp = np.array([logp(xi) for xi in x])
    chain = np.empty((nsteps, w, d))
    lps = np.empty((nsteps, w))
    acc = 0
    half = w // 2
    for t in range(nsteps):
        for s in (0, 1):
            idx = np.arange(s * half, (s + 1) * half)
            oth = np.arange((1 - s) * half, (2 - s) * half)
            partner = x[rng.choice(oth, size=half)]
            z = ((a - 1.0) * rng.random(half) + 1.0) ** 2 / a
            prop = partner + z[:, None] * (x[idx] - partner)
            lp_new = np.array([logp(p) for p in prop])
            with np.errstate(invalid="ignore"):
                log_acc = (d - 1.0) * np.log(z) + lp_new - lp[idx]
            ok = np.log(rng.random(half)) < log_acc
            ok &= np.isfinite(lp_new)
            x[idx[ok]] = prop[ok]
            lp[idx[ok]] = lp_new[ok]
            acc += int(ok.sum())
        chain[t], lps[t] = x, lp
    return chain, lps, acc / (nsteps * w)


def sample(post: Posterior, x0: np.ndarray, nwalkers: int | None = None, nsteps: int = 400, burn: int = 200,
           seed: int = 0, backend: str = "auto", init_scale: float = 1e-2) -> Chain:
    """Run the ensemble sampler around ``x0`` (use the LSQ solution). ``backend``: 'emcee', 'stretch' or 'auto'."""
    d = post.dim
    nwalkers = nwalkers or max(2 * d + 2, 16)  # magic: minimum ensemble size
    nwalkers += nwalkers % 2
    rng = np.random.default_rng(seed)
    p0 = x0[None, :] + init_scale * rng.standard_normal((nwalkers, d))
    use = backend
    if backend == "auto":
        try:
            import emcee  # noqa: F401

            use = "emcee"
        except Exception:
            use = "stretch"
    if use == "emcee":
        import emcee

        sam = emcee.EnsembleSampler(nwalkers, d, post)
        sam.run_mcmc(p0, nsteps, progress=False)
        ch, lp, acc = sam.get_chain(), sam.get_log_prob(), float(np.mean(sam.acceptance_fraction))
        try:
            tau = float(np.mean(sam.get_autocorr_time(tol=0)))
        except Exception:
            tau = float("nan")
    else:
        ch, lp, acc = stretch_move(post, p0, nsteps, seed=seed)
        tau = float(np.mean([autocorr_time(ch[burn:, :, k].mean(axis=1)) for k in range(d)])) if nsteps > burn + 10 else float("nan")  # magic: min length
    chain = Chain(ch[burn:], lp[burn:], acc, use, post)
    chain.diag = {"autocorr_time": tau, "acceptance": acc, "ess_est": float(chain.samples.shape[0] * chain.samples.shape[1] / max(tau, 1.0)),
                  "rhat": split_rhat(chain.samples)}
    return chain


def autocorr_time(x: np.ndarray) -> float:
    """Integrated autocorrelation time of a 1-D series (Sokal windowing)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf = acf / acf[0] if acf[0] > 0 else acf
    tau = 1.0 + 2.0 * np.cumsum(acf[1:])
    m = np.arange(1, n) < 5.0 * tau  # magic: Sokal window constant
    k = int(np.argmin(m)) if not m.all() else n - 1
    return float(tau[max(k - 1, 0)])


def split_rhat(samples: np.ndarray) -> np.ndarray:
    """Split-R-hat per dimension from an array (nsteps, nchains, dim): chains are split in half."""
    n = samples.shape[0] // 2
    if n < 4:  # magic: too short
        return np.full(samples.shape[-1], np.nan)
    ch = np.concatenate([samples[:n], samples[n: 2 * n]], axis=1)
    m = ch.shape[1]
    mean_c = ch.mean(axis=0)
    w = ch.var(axis=0, ddof=1).mean(axis=0)
    b = n * mean_c.var(axis=0, ddof=1)
    var_hat = (n - 1) / n * w + b / n
    _ = m
    return np.sqrt(var_hat / np.maximum(w, 1e-300))  # magic: floor


# ---------------------------------------------------------------- summaries
def credible_intervals(chain: Chain, level: float = 0.95, thin: int = 1) -> dict[str, dict[str, float]]:
    nat = chain.natural(thin)
    lo, hi = (1 - level) / 2, 1 - (1 - level) / 2
    return {n: {"median": float(np.median(v)), "lo": float(np.quantile(v, lo)), "hi": float(np.quantile(v, hi)),
                "mean": float(v.mean()), "sd": float(v.std())} for n, v in nat.items()}


def posterior_predictive(chain: Chain, n_draw: int = 100, seed: int = 0, with_noise: bool = True
                         ) -> list[list[dict[str, np.ndarray]]]:
    """Posterior-predictive bands per experiment/channel: {'lo','med','hi','pred_lo','pred_hi'} (95 %)."""
    rng = np.random.default_rng(seed)
    post = chain.posterior
    fl = chain.flat()
    pick = fl[rng.choice(len(fl), size=min(n_draw, len(fl)), replace=False)]
    draws: list[list[list[np.ndarray]]] = []
    for z in pick:
        x, s = post.split(z)
        draws.append(post.pred.predict(post.pred.base.with_(**post.space.from_u(x))))
    out = []
    for ie, e in enumerate(post.pred.exps):
        row = []
        for ic, c in enumerate(e.channels):
            arr = np.array([d[ie][ic] for d in draws])
            sig = c.sigma if with_noise else 0.0
            noisy = arr + sig * rng.standard_normal(arr.shape) if with_noise else arr
            row.append({"t": c.t, "lo": np.quantile(arr, 0.025, axis=0), "med": np.median(arr, axis=0),
                        "hi": np.quantile(arr, 0.975, axis=0), "pred_lo": np.quantile(noisy, 0.025, axis=0),
                        "pred_hi": np.quantile(noisy, 0.975, axis=0)})
        out.append(row)
    return out


def ppc_coverage(chain: Chain, n_draw: int = 100, seed: int = 0) -> dict[str, float]:
    """Posterior predictive check: fraction of observations inside the 95 % predictive band."""
    bands = posterior_predictive(chain, n_draw, seed)
    inside = tot = 0
    for e, row in zip(chain.posterior.pred.exps, bands, strict=True):
        for c, b in zip(e.channels, row, strict=True):
            inside += int(np.sum((c.y >= b["pred_lo"]) & (c.y <= b["pred_hi"])))
            tot += len(c.y)
    return {"coverage95": inside / tot, "n": tot}


def waic(chain: Chain, n_draw: int = 200, seed: int = 0) -> dict[str, float]:
    """WAIC (log pointwise predictive density minus effective parameters)."""
    rng = np.random.default_rng(seed)
    fl = chain.flat()
    pick = fl[rng.choice(len(fl), size=min(n_draw, len(fl)), replace=False)]
    ll = np.array([chain.posterior.loglik_terms(z) for z in pick])  # (S, N)
    m = ll.max(axis=0)
    lppd = np.sum(m + np.log(np.mean(np.exp(ll - m), axis=0)))
    p_waic = float(np.sum(ll.var(axis=0, ddof=1)))
    return {"waic": float(-2.0 * (lppd - p_waic)), "lppd": float(lppd), "p_waic": p_waic}


def corner_data(chain: Chain, bins: int = 30, thin: int = 1) -> dict[str, Any]:
    """Pairwise 2-D histograms and marginals of the physical parameters (for corner plots)."""
    nat = chain.natural(thin)
    names = list(nat)
    marg = {n: np.histogram(nat[n], bins=bins, density=True) for n in names}
    pair = {}
    for i, a in enumerate(names):
        for b in names[:i]:
            h, xe, ye = np.histogram2d(nat[b], nat[a], bins=bins)
            pair[(a, b)] = (h, xe, ye)
    return {"names": names, "marginals": marg, "pairs": pair, "corr": np.corrcoef(np.array([nat[n] for n in names]))}


def plot_corner(chain: Chain, path: str, bins: int = 30) -> str:
    """Write a corner plot PNG (matplotlib, offline)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cd = corner_data(chain, bins)
    names = cd["names"]
    k = len(names)
    fig, ax = plt.subplots(k, k, figsize=(2.2 * k, 2.2 * k))
    ax = np.atleast_2d(ax)
    nat = chain.natural()
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            axx = ax[i, j]
            if j > i:
                axx.axis("off")
            elif i == j:
                axx.hist(nat[a], bins=bins, color="#4c78a8")
                axx.set_yticks([])
            else:
                axx.hist2d(nat[b], nat[a], bins=bins, cmap="Blues")
            if i == k - 1:
                axx.set_xlabel(b if j < k else a)
            if j == 0 and i > 0:
                axx.set_ylabel(a)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
