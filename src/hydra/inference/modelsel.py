"""Model selection among the empirical, mass-transfer-limited and electrochemical (reduced) rate laws."""

from __future__ import annotations

from typing import Any

from ..core.params import ParamSet
from .bayes import Posterior, sample, waic
from .calibrate import FitResult, Predictor, fit_lsq
from .data import Experiment

MODEL_PARAMS = {
    "empirical": ["k25", "Ea", "n_oh", "tau_ind"],
    "mass_transfer": ["k25", "Ea", "n_oh", "tau_ind", "k_mt"],
    "echem": ["k25", "Ea", "n_oh", "ec_k_diss", "ec_k_pass", "ec_eps_al"],
}


def compare_models(exps: list[Experiment], models: list[str] | None = None, base: ParamSet | None = None,
                   dt: float = 2.0, with_waic: bool = False, nsteps: int = 200, burn: int = 100,
                   seed: int = 0) -> list[dict[str, Any]]:
    """Fit each model; report chi2, AIC, BIC (and WAIC from MCMC if requested), ranked by BIC.

    Delta-AIC/BIC > 10 means essentially no support for the worse model; honest caveat: all inferences are
    conditional on the same data and on the sigmas supplied with them."""
    base = base or ParamSet()
    rows = []
    for m in models or list(MODEL_PARAMS):
        pred = Predictor(exps, base, m, dt)
        fit: FitResult = fit_lsq(pred, MODEL_PARAMS[m])
        row: dict[str, Any] = {"model": m, "k": len(fit.names), "chi2": fit.chi2, "red_chi2": fit.red_chi2,
                               "aic": fit.aic, "bic": fit.bic, "theta": fit.theta}
        if with_waic:
            post = Posterior(pred, MODEL_PARAMS[m])
            ch = sample(post, post.start(fit), nsteps=nsteps, burn=burn, seed=seed, backend="stretch")
            row["waic"] = waic(ch, 100)["waic"]  # magic: draws
        rows.append(row)
    best_bic = min(r["bic"] for r in rows)
    best_aic = min(r["aic"] for r in rows)
    for r in rows:
        r["dBIC"], r["dAIC"] = r["bic"] - best_bic, r["aic"] - best_aic
    return sorted(rows, key=lambda r: r["bic"])
