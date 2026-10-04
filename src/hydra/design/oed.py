r"""Optimal experimental design: "what experiment should we run next?".

Given the current posterior (covariance :math:`\Sigma` of the decorrelated coordinates ``x``) and lab constraints,
rank candidate experiments by expected information gain (EIG):

.. math:: \mathrm{EIG}(d)=\tfrac12\log\det\!\big(I+\Sigma J_d^{T}W_dJ_d\big)\quad(\text{linear-Gaussian approximation}),

with :math:`J_d=\partial\eta_d/\partial x` and :math:`W_d=\mathrm{diag}(1/\sigma^2)` over the planned
measurements; ``focus`` restricts the gain to the most uncertain parameters via the posterior-covariance
Schur complement. A nested Monte-Carlo EIG (:func:`eig_mc`) is provided to check the linearisation.
The output is a printable lab protocol (quantities, expected curve with bands, stop conditions).
"""

from __future__ import annotations

import html
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from ..constants import MW_NAOH
from ..core import l1_fast as lf
from ..core import safety
from ..core.params import ParamSet
from ..core.scenario import Scenario
from ..inference.calibrate import FitResult
from ..inference.data import ParamSpace
from ..inference.sensors import Sensor
from ..thermo import propdb as DB


class LabConstraints(BaseModel):
    forms: list[str] = Field(default_factory=lambda: ["foil", "powder"])
    dim_um_by_form: dict[str, list[float]] = Field(default_factory=lambda: {"foil": [10, 20, 50], "powder": [50, 100, 250],
                                                                         "wire": [200, 500], "can": [100]})
    c_naoh_max_M: float = 4.0
    c_naoh_min_M: float = 0.2
    T0_range_C: tuple[float, float] = (15.0, 45.0)
    al_mass_range_g: tuple[float, float] = (0.2, 3.0)
    v_liq_choices_mL: list[float] = Field(default_factory=lambda: [100.0, 200.0])
    v_vessel_mL: float = 500.0
    max_duration_s: float = 3600.0
    peak_T_limit_C: float = 70.0  # liquid temperature stop limit
    n_h2_points: int = 12
    n_T_points: int = 12
    sigma_n_h2: float = 3.0e-4  # mol
    sigma_T: float = 0.2  # K
    T_sensor_tau_s: float = 8.0
    min_conversion: float = 0.3  # the experiment must see this much conversion within max_duration
    channels: list[str] = Field(default_factory=lambda: ["n_h2", "T"])


@dataclass
class Candidate:
    scenario: Scenario
    eig: float = 0.0
    eig_focus: float = 0.0
    peak_T_C: float = 0.0
    conversion: float = 0.0
    why_not: list[str] = field(default_factory=list)


def _schedule(duration: float, n: int) -> np.ndarray:
    """Sampling times: denser early (induction + ramp) then uniform."""
    early = np.geomspace(duration / 100.0, duration / 4.0, n // 2)  # magic: first quarter log-spaced
    late = np.linspace(duration / 4.0, duration, n - n // 2 + 1)[1:]
    return np.unique(np.concatenate([early, late]))


def _predict(sc: Scenario, p: ParamSet, cons: LabConstraints, model: str, dt: float) -> dict[str, np.ndarray]:
    return lf.simulate_fast(lf.build_constants(sc, p, model, dt), sc, dt)


def _obs_vector(sim: dict[str, np.ndarray], sc: Scenario, cons: LabConstraints) -> tuple[np.ndarray, np.ndarray]:
    """Predicted measurement vector and its sigmas for the planned channels."""
    out, sig = [], []
    if "n_h2" in cons.channels:
        t = _schedule(sc.duration_s, cons.n_h2_points)
        out.append(np.interp(t, sim["t"], sim["gen"]))
        sig.append(np.full(len(t), cons.sigma_n_h2))
    if "T" in cons.channels:
        t = _schedule(sc.duration_s, cons.n_T_points)
        s = Sensor(tau_s=cons.T_sensor_tau_s)
        out.append(np.interp(t, sim["t"], s.apply(sim["t"], sim["T"])))
        sig.append(np.full(len(t), cons.sigma_T))
    return np.concatenate(out), np.concatenate(sig)


def jacobian_x(sc: Scenario, base: ParamSet, space: ParamSpace, x_hat: np.ndarray, cons: LabConstraints,
               model: str = "empirical", dt: float = 4.0, h: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """d(measurement vector)/dx at x_hat by central differences (x are decorrelated O(1) coordinates)."""
    y0, sig = _obs_vector(_predict(sc, base.with_(**space.from_u(x_hat)), cons, model, dt), sc, cons)
    j = np.zeros((len(y0), len(x_hat)))
    for i in range(len(x_hat)):
        xp, xm = x_hat.copy(), x_hat.copy()
        xp[i] += h
        xm[i] -= h
        yp = _obs_vector(_predict(sc, base.with_(**space.from_u(xp)), cons, model, dt), sc, cons)[0]
        ym = _obs_vector(_predict(sc, base.with_(**space.from_u(xm)), cons, model, dt), sc, cons)[0]
        j[:, i] = (yp - ym) / (2 * h)
    return j, sig


def eig_linear(j: np.ndarray, sig: np.ndarray, cov_x: np.ndarray, focus: list[int] | None = None) -> float:
    """Linearised EIG (nats). With ``focus``: information gained on those parameters only."""
    jw = j / sig[:, None]
    fim = jw.T @ jw
    n = cov_x.shape[0]
    if focus is None:
        return float(0.5 * np.linalg.slogdet(np.eye(n) + cov_x @ fim)[1])
    post = np.linalg.inv(np.linalg.inv(cov_x) + fim)
    f = np.array(focus)
    return float(0.5 * (np.linalg.slogdet(cov_x[np.ix_(f, f)])[1] - np.linalg.slogdet(post[np.ix_(f, f)])[1]))


def eig_mc(sc: Scenario, base: ParamSet, space: ParamSpace, x_samples: np.ndarray, cons: LabConstraints,
           n_outer: int = 60, n_inner: int = 120, model: str = "empirical", dt: float = 4.0, seed: int = 0) -> float:
    """Nested Monte-Carlo EIG using posterior samples (Gaussian measurement noise)."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(x_samples), size=n_outer + n_inner, replace=len(x_samples) < n_outer + n_inner)
    preds = np.array([_obs_vector(_predict(sc, base.with_(**space.from_u(x_samples[i])), cons, model, dt), sc, cons)[0]
                      for i in idx])
    sig = _obs_vector(_predict(sc, base, cons, model, dt), sc, cons)[1]
    outer, inner = preds[:n_outer], preds[n_outer:]
    out = 0.0
    for m in outer:
        y = m + sig * rng.standard_normal(len(m))
        ll_true = -0.5 * np.sum(((y - m) / sig) ** 2)
        ll_in = -0.5 * np.sum(((y[None, :] - inner) / sig) ** 2, axis=1)
        mx = ll_in.max()
        out += ll_true - (mx + np.log(np.mean(np.exp(ll_in - mx))))
    return float(out / n_outer)


def generate_candidates(cons: LabConstraints, n: int = 200, seed: int = 0, base: ParamSet | None = None,
                        model: str = "empirical") -> list[Candidate]:
    """Random feasible candidate experiments (within lab constraints, screened for safety and measurability)."""
    rng = np.random.default_rng(seed)
    base = base or ParamSet()
    out: list[Candidate] = []
    tries = 0
    while len(out) < n and tries < 20 * n:  # magic: sampling budget
        tries += 1
        form = str(rng.choice(cons.forms))
        sc = Scenario(
            al_mass_g=float(rng.uniform(*cons.al_mass_range_g)), form=form,  # type: ignore[arg-type]
            dim_um=float(rng.choice(cons.dim_um_by_form[form])),
            c_naoh_M=float(rng.uniform(cons.c_naoh_min_M, cons.c_naoh_max_M)),
            v_liq_mL=float(rng.choice(cons.v_liq_choices_mL)), v_vessel_mL=cons.v_vessel_mL,
            T0_C=float(rng.uniform(*cons.T0_range_C)), T_amb_C=float(rng.uniform(18.0, 28.0)),  # magic: lab ambient range
            mode="open", duration_s=cons.max_duration_s, vent_diameter_mm=10.0)
        why = []
        rep = safety.screen(sc, base)
        if rep.unsafe:
            why.append("pre-run safety screen: " + "; ".join(f.code for f in rep.flags if f.level == "danger"))
        sim = _predict(sc, base, cons, model, 4.0)
        peak = float(sim["T"].max() - 273.15)
        conv = float(sim["gen"][-1] / (1.5 * lf.build_constants(sc, base)[lf.KI["n0"]]))
        if peak > cons.peak_T_limit_C:
            why.append(f"peak T {peak:.0f} C > limit {cons.peak_T_limit_C:.0f} C")
        if conv < cons.min_conversion:
            why.append(f"only {100 * conv:.0f}% conversion within {cons.max_duration_s:.0f} s")
        if not why:
            out.append(Candidate(sc, peak_T_C=peak, conversion=conv))
    return out


def rank_candidates(cands: list[Candidate], fit: FitResult, cons: LabConstraints, base: ParamSet | None = None,
                    focus_names: list[str] | None = None, model: str = "empirical", cov_x: np.ndarray | None = None,
                    dt: float = 4.0) -> list[Candidate]:
    """Compute EIG for every candidate and sort (best first). Focus defaults to the most uncertain parameters."""
    base = base or ParamSet()
    space = fit.space
    x_hat = space.to_u(fit.theta)
    cov = cov_x if cov_x is not None else fit.cov
    if focus_names is None:
        order = np.argsort(-np.diag(cov))  # largest posterior variance (in O(1) x units)
        focus = [int(i) for i in order[: max(2, len(order) // 2)]]
    else:
        focus = [fit.names.index(n) for n in focus_names]
    for c in cands:
        j, sig = jacobian_x(c.scenario, base, space, x_hat, cons, model, dt)
        c.eig = eig_linear(j, sig, cov)
        c.eig_focus = eig_linear(j, sig, cov, focus)
    return sorted(cands, key=lambda c: -c.eig_focus)


# ---------------------------------------------------------------- printable protocol
@dataclass
class Protocol:
    scenario: Scenario
    quantities: dict[str, float]
    schedule_s: list[float]
    expected: dict[str, list[float]]
    stop_conditions: list[str]
    safety: list[str]
    eig: float
    note: str = ""

    def to_markdown(self) -> str:
        q = self.quantities
        lines = [f"# Next experiment (EIG = {self.eig:.2f} nats)", "", "## Quantities",
                 f"- Aluminium: **{q['al_mass_g']:.2f} g** {self.scenario.form}, characteristic size {self.scenario.dim_um:.0f} um",
                 f"- NaOH: **{q['naoh_g']:.2f} g** pellets to make **{q['v_liq_mL']:.0f} mL** of {q['c_naoh_M']:.2f} M solution "
                 f"(add pellets to ~{0.8 * q['v_liq_mL']:.0f} mL water slowly - strongly exothermic - then top up, "
                 f"let it cool to {self.scenario.T0_C:.0f} C before adding the aluminium)",
                 f"- Vessel: {self.scenario.v_vessel_mL:.0f} mL, open/vented to a gas-collection line",
                 f"- Start temperature: {self.scenario.T0_C:.0f} C (ambient {self.scenario.T_amb_C:.0f} C)",
                 f"- Planned duration: {self.scenario.duration_s / 60:.0f} min", "", "## Measurements (log at these times, s)",
                 ", ".join(f"{t:.0f}" for t in self.schedule_s), "", "## Expected curve (model median and 95 % band)",
                 "| t [s] | n_H2 [mmol] | lo | hi | T [C] |", "|---|---|---|---|---|"]
        for i, t in enumerate(self.expected["t"]):
            lines.append(f"| {t:.0f} | {1e3 * self.expected['n_med'][i]:.2f} | {1e3 * self.expected['n_lo'][i]:.2f} | "
                         f"{1e3 * self.expected['n_hi'][i]:.2f} | {self.expected['T_med'][i] - 273.15:.1f} |")
        lines += ["", "## Stop conditions"] + [f"- {s}" for s in self.stop_conditions]
        lines += ["", "## Safety"] + [f"- {s}" for s in self.safety]
        if self.note:
            lines += ["", self.note]
        return "\n".join(lines)

    def to_html(self) -> str:
        return ("<html><head><meta charset='utf-8'><title>HYDRA protocol</title><style>body{font:14px sans-serif;max-width:800px;"
                "margin:2em auto}</style></head><body><pre style='white-space:pre-wrap'>" + html.escape(self.to_markdown())
                + "</pre></body></html>")


def make_protocol(cand: Candidate, fit: FitResult, cons: LabConstraints, base: ParamSet | None = None, n_draw: int = 100,
                  model: str = "empirical", dt: float = 2.0, seed: int = 0, x_samples: np.ndarray | None = None) -> Protocol:
    base = base or ParamSet()
    sc = cand.scenario
    rng = np.random.default_rng(seed)
    if x_samples is None:
        x_samples = rng.multivariate_normal(fit.space.to_u(fit.theta), fit.cov, size=n_draw)
    sims = [_predict(sc, base.with_(**fit.space.from_u(x)), cons, model, dt) for x in x_samples]
    t = np.linspace(0.0, sc.duration_s, 25)
    n = np.array([np.interp(t, s["t"], s["gen"]) for s in sims])
    tt = np.array([np.interp(t, s["t"], s["T"]) for s in sims])
    exp = {"t": t.tolist(), "n_med": np.median(n, 0).tolist(), "n_lo": np.quantile(n, 0.025, 0).tolist(),
           "n_hi": np.quantile(n, 0.975, 0).tolist(), "T_med": np.median(tt, 0).tolist()}
    n_mol = sc.c_naoh_M * sc.v_liq_mL / 1e3
    peak_hi = float(np.quantile([s["T"].max() for s in sims], 0.975)) - 273.15
    t_limit = min(cons.peak_T_limit_C, base["T_hdpe_max"] - 273.15 - 10.0)  # magic: margin below HDPE service limit
    stops = [f"Liquid temperature reaches {t_limit:.0f} C (model 97.5 % peak: {peak_hi:.0f} C): remove heat source, add cold water bath, abort.",
             "Any boiling, foaming up to the vessel neck, or liquid/mist carry-over into the gas line.",
             "Gas flow falls below 2 % of its peak after 90 % of the expected hydrogen has been collected (reaction finished).",
             f"Elapsed time exceeds {sc.duration_s / 60:.0f} min.",
             "Vessel swelling, hissing from a closure, or unexpected pressure rise on the gauge: stop and vent."]
    rep = safety.screen(sc, base)
    safe = [f.message for f in rep.flags if f.code != "DISCLAIMER"] + [safety.DISCLAIMER]
    return Protocol(sc, {"al_mass_g": sc.al_mass_g, "naoh_g": n_mol * MW_NAOH * 1e3, "v_liq_mL": sc.v_liq_mL,
                         "c_naoh_M": sc.c_naoh_M}, _schedule(sc.duration_s, cons.n_h2_points).tolist(), exp, stops, safe,
                    cand.eig_focus or cand.eig, "Model is uncalibrated outside the data it was fitted to; treat bands as a guide.")


_ = DB
