"""Data reconciliation with gross-error detection (volumetric vs mass-loss H2 measurements)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import chi2, norm


@dataclass
class Reconciled:
    estimate: float
    sd: float
    chi2: float
    dof: int
    p_value: float
    flagged: list[int]
    residuals: np.ndarray  # normalised residuals of the final estimate (all inputs)


def reconcile(values: np.ndarray, sigmas: np.ndarray, alpha: float = 0.05) -> Reconciled:
    """Weighted consensus of redundant measurements of one quantity with iterative gross-error removal.

    Global test: chi2 of the weighted residuals vs the chi2(k-1) quantile; measurement test: normalised
    residual vs a Bonferroni-corrected normal quantile. The worst offender is removed until both pass."""
    v = np.asarray(values, float)
    s = np.asarray(sigmas, float)
    keep = list(range(len(v)))
    flagged: list[int] = []
    while True:
        w = 1.0 / s[keep] ** 2
        est = float(np.sum(w * v[keep]) / np.sum(w))
        sd = float(np.sqrt(1.0 / np.sum(w)))
        r = (v[keep] - est) / np.sqrt(s[keep] ** 2 - sd**2 + 1e-30)  # magic: floor; residual variance
        c2 = float(np.sum(((v[keep] - est) / s[keep]) ** 2))
        dof = len(keep) - 1
        p = float(1.0 - chi2.cdf(c2, dof)) if dof > 0 else 1.0
        zcrit = norm.ppf(1.0 - alpha / (2.0 * max(len(keep), 1)))
        if (p >= alpha and np.all(np.abs(r) < zcrit)) or len(keep) <= 2:  # magic: need >=2 to cross-check
            break
        worst = keep[int(np.argmax(np.abs(r)))]
        flagged.append(worst)
        keep.remove(worst)
    full_r = (v - est) / np.sqrt(s**2 + 1e-30)  # magic: floor
    return Reconciled(est, sd, c2, dof, p, flagged, full_r)


def reconcile_curves(t_ref: np.ndarray, curves: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
                     alpha: float = 0.05) -> dict[str, np.ndarray]:
    """Point-wise reconciliation of cumulative-mole curves from several methods on a common time axis.

    Each curve is ``(t, n, sigma_n)``. Returns the consensus curve and the number of flagged methods per point."""
    est = np.zeros(len(t_ref))
    sd = np.zeros(len(t_ref))
    nflag = np.zeros(len(t_ref), int)
    for i, t in enumerate(t_ref):
        vals = np.array([np.interp(t, c[0], c[1]) for c in curves])
        sig = np.array([np.interp(t, c[0], c[2]) for c in curves])
        rec = reconcile(vals, np.maximum(sig, 1e-12), alpha)  # magic: floor
        est[i], sd[i], nflag[i] = rec.estimate, rec.sd, len(rec.flagged)
    return {"t": t_ref, "n": est, "sd": sd, "n_flagged": nflag}


def bias_factor(n_a: np.ndarray, n_b: np.ndarray, sig_a: np.ndarray, sig_b: np.ndarray) -> tuple[float, float]:
    """Systematic scale factor beta in n_b = beta n_a (regression through origin, errors in both)."""
    w = 1.0 / (sig_b**2 + (sig_a) ** 2)
    beta = float(np.sum(w * n_a * n_b) / np.sum(w * n_a * n_a))
    se = float(np.sqrt(1.0 / np.sum(w * n_a * n_a)))
    return beta, se
