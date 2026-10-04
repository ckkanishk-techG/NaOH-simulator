r"""Live digital twin: Extended and Ensemble Kalman filters on the reduced L1 model with online parameter estimation.

Augmented state ``x = [f, n_AlO, T, T_w, psi, G, T_s, ln m_k, ln m_R]``: Al fraction left, aluminate, liquid/wall
temperature, film, cumulative H2, **thermocouple state** (first-order lag, so sensor delay is modelled, not
ignored), a log rate-constant multiplier ``m_k`` (random walk) and a log membrane-resistance multiplier ``m_R``
(random walk, observed through the stack voltage at known current). Rolling forecasts propagate the filter
distribution with the model; early warnings come from the same limits as the safety engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..constants import KELVIN_OFFSET, F
from ..core import l1_fast as lf
from ..core.params import ParamSet
from ..core.scenario import Scenario
from ..fuelcell.stack import Stack, StackSpec
from ..thermo import propdb as DB
from .sources import Frame

NS = 9
I_F, I_AL, I_T, I_TW, I_PSI, I_G, I_TS, I_LK, I_LR = range(NS)


@dataclass
class TwinConfig:
    method: str = "enkf"  # 'ekf' | 'enkf'
    n_ens: int = 60
    dt_model: float = 2.0
    tau_tc: float = 8.0  # thermocouple time constant [s]
    sd_T: float = 0.15  # K
    sd_gen: float = 3.0e-4  # mol (cumulative-H2 sensor)
    sd_V: float = 0.01  # V (stack)
    q_lnk: float = 2.0e-3  # random-walk sd of ln m_k per sqrt(s)
    q_lnr: float = 1.0e-3
    init_sd_lnk: float = 0.5
    init_sd_lnr: float = 0.3
    inflation: float = 1.02
    limit_T_C: float = 70.0
    n_cells: int = 4
    seed: int = 0


@dataclass
class Forecast:
    t: np.ndarray
    T_lo: np.ndarray
    T_med: np.ndarray
    T_hi: np.ndarray
    gen_lo: np.ndarray
    gen_med: np.ndarray
    gen_hi: np.ndarray
    p_exceed_T: float
    t_to_limit_s: dict[str, float]


@dataclass
class Warning_:
    level: str
    code: str
    message: str


class LiveTwin:
    def __init__(self, sc: Scenario, base: ParamSet | None = None, cfg: TwinConfig | None = None) -> None:
        self.sc, self.base, self.cfg = sc, base or ParamSet(), cfg or TwinConfig()
        self.k0 = lf.build_constants(sc, self.base, "empirical", self.cfg.dt_model)
        self.rng = np.random.default_rng(self.cfg.seed)
        T0 = sc.T0_C + KELVIN_OFFSET
        self.x0 = np.array([1.0, 0.0, T0, T0, 0.0, 0.0, T0, 0.0, 0.0])
        c = self.cfg
        self.P0 = np.diag([1e-6, 1e-8, 0.05, 0.05, 1e-4, 1e-8, 0.05, c.init_sd_lnk**2, c.init_sd_lnr**2])
        self.stack = Stack(StackSpec(n_series=c.n_cells, seed=0), self.base)
        self.t = 0.0
        self.history: list[dict[str, float]] = []
        self.reset()

    # ------------------------------------------------------------------ model
    def _advance(self, x: np.ndarray, dt: float) -> np.ndarray:
        """Process model over ``dt`` seconds (reduced model sub-stepped at dt_model)."""
        n = max(int(round(dt / self.cfg.dt_model)), 1)
        h = dt / n
        k = self.k0.copy()
        k[lf.KI["k25"]] *= math.exp(x[I_LK])
        out = lf._NP_INT(np.ascontiguousarray(x[:6]), k, h, n)[-1]
        xn = x.copy()
        xn[:6] = out
        a = math.exp(-dt / self.cfg.tau_tc)
        xn[I_TS] = a * x[I_TS] + (1.0 - a) * 0.5 * (x[I_T] + out[I_T])  # exponential lag of the (mean) true temperature
        return xn

    def _h(self, x: np.ndarray, frame: Frame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predicted observations, their sigmas and indices for the channels present in ``frame``."""
        c = self.cfg
        pred, sig, idx = [], [], []
        if math.isfinite(frame.T1):
            pred.append(x[I_TS] - KELVIN_OFFSET)
            sig.append(c.sd_T)
            idx.append(0)
        if math.isfinite(frame.T2):
            pred.append(x[I_TW] - KELVIN_OFFSET)
            sig.append(c.sd_T)
            idx.append(1)
        if math.isfinite(frame.flow):  # integrated H2 -> use the cumulative-moles channel (1 Hz flow integrates outside)
            pass
        if math.isfinite(frame.V) and math.isfinite(frame.I) and frame.I > 0:
            sp = ParamSet({"fc_asr": self.base["fc_asr"] * math.exp(x[I_LR])})
            st = Stack(StackSpec(n_series=c.n_cells, seed=0), sp)
            pred.append(st.voltage(frame.I))
            sig.append(c.sd_V)
            idx.append(2)
        return np.array(pred), np.array(sig), np.array(idx)

    def _obs_vector(self, frame: Frame, n_prev: float | None) -> np.ndarray:
        out = []
        if math.isfinite(frame.T1):
            out.append(frame.T1)
        if math.isfinite(frame.T2):
            out.append(frame.T2)
        if math.isfinite(frame.V) and math.isfinite(frame.I) and frame.I > 0:
            out.append(frame.V)
        _ = n_prev
        return np.array(out)

    # ------------------------------------------------------------------ filters
    def reset(self) -> None:
        c = self.cfg
        self.t = 0.0
        self.x = self.x0.copy()
        self.P = self.P0.copy()
        self.ens = self.x0[None, :] + self.rng.multivariate_normal(np.zeros(NS), np.diag(np.diag(self.P0)), c.n_ens)
        self.ens[:, [I_F, I_AL, I_G]] = self.x0[[I_F, I_AL, I_G]]  # invariants exactly known at t=0
        self.gen_obs: float | None = None
        self.best_k: tuple[float, float] | None = None
        self.history.clear()

    def _q(self, dt: float) -> np.ndarray:
        c = self.cfg
        q = np.zeros(NS)
        q[[I_T, I_TW, I_TS]] = (0.002 ** 2) * dt
        active = min(1.0, max(float(self.x[I_F]), 0.0) / 0.1)  # magic: the rate constant is unobservable once the Al is gone
        q[I_LK] = (c.q_lnk**2) * dt * active
        q[I_LR] = c.q_lnr**2 * dt
        return np.diag(q)

    def ingest(self, frame: Frame, cumulative_h2_mol: float | None = None) -> dict[str, float]:
        """Predict to the frame time and assimilate whatever channels it carries. Out-of-order frames are ignored."""
        dt = frame.t - self.t
        if dt <= 0.0:
            return self.status()
        c = self.cfg
        if c.method == "ekf":
            self._ekf(frame, dt, cumulative_h2_mol)
        else:
            self._enkf(frame, dt, cumulative_h2_mol)
        self.t = frame.t
        s = self.status()
        if self.best_k is None or s["k_mult_sd_ln"] < self.best_k[1]:
            self.best_k = (s["k_mult"], s["k_mult_sd_ln"])  # most informative estimate of the rate multiplier
        s = self.status()
        self.history.append(s)
        return s

    def _extra_obs(self, cum: float | None, sig: list[float], idx_list: list[int]) -> None:
        _ = (cum, sig, idx_list)

    def _assemble(self, x: np.ndarray, frame: Frame, cum: float | None) -> tuple[np.ndarray, np.ndarray]:
        pred, sig, _ = self._h(x, frame)
        if cum is not None and math.isfinite(cum):
            pred = np.append(pred, x[I_G])
            sig = np.append(sig, self.cfg.sd_gen)
        return pred, sig

    def _y(self, frame: Frame, cum: float | None) -> np.ndarray:
        y = self._obs_vector(frame, None)
        if cum is not None and math.isfinite(cum):
            y = np.append(y, cum)
        return y

    def _project(self, x: np.ndarray) -> np.ndarray:
        """Re-impose the model invariants (Al conservation): f = 1 - G/(1.5 n0), n_AlO = n0 (1-f); bound the multipliers."""
        n0 = self.k0[lf.KI["n0"]]
        x = np.atleast_2d(x).copy()
        x[:, I_G] = np.clip(x[:, I_G], 0.0, 1.5 * n0)
        x[:, I_F] = 1.0 - x[:, I_G] / (1.5 * n0)
        x[:, I_AL] = n0 * (1.0 - x[:, I_F])
        x[:, I_LK] = np.clip(x[:, I_LK], math.log(0.1), math.log(10.0))  # magic: plausible rate-constant range
        x[:, I_LR] = np.clip(x[:, I_LR], math.log(0.2), math.log(10.0))  # magic: plausible resistance range
        return x

    def _ekf(self, frame: Frame, dt: float, cum: float | None) -> None:
        xp = self._advance(self.x, dt)
        # numerical Jacobian of the process model
        jac = np.eye(NS)
        for i in range(NS):
            h = 1e-6 * max(abs(self.x[i]), 1.0) if i not in (I_LK, I_LR) else 1e-4
            xa, xb = self.x.copy(), self.x.copy()
            xa[i] += h
            xb[i] -= h
            jac[:, i] = (self._advance(xa, dt) - self._advance(xb, dt)) / (2 * h)
        pp = jac @ self.P @ jac.T + self._q(dt)
        y = self._y(frame, cum)
        if len(y):
            hx, sig = self._assemble(xp, frame, cum)
            hm = np.zeros((len(hx), NS))
            for i in range(NS):
                h = 1e-6 * max(abs(xp[i]), 1.0) if i not in (I_LK, I_LR) else 1e-4
                xa, xb = xp.copy(), xp.copy()
                xa[i] += h
                xb[i] -= h
                hm[:, i] = (self._assemble(xa, frame, cum)[0] - self._assemble(xb, frame, cum)[0]) / (2 * h)
            r = np.diag(sig**2)
            s_ = hm @ pp @ hm.T + r
            kg = pp @ hm.T @ np.linalg.inv(s_)
            xp = xp + kg @ (y - hx)
            pp = (np.eye(NS) - kg @ hm) @ pp
        xp = self._project(xp)[0]
        self.x, self.P = xp, 0.5 * (pp + pp.T)

    def _enkf(self, frame: Frame, dt: float, cum: float | None) -> None:
        c = self.cfg
        n = c.n_ens
        q_sd = np.sqrt(np.diag(self._q(dt)))
        ens = np.array([self._advance(m, dt) for m in self.ens]) + q_sd[None, :] * self.rng.standard_normal((n, NS))
        y = self._y(frame, cum)
        if len(y):
            hx = np.array([self._assemble(m, frame, cum)[0] for m in ens])
            sig = self._assemble(ens[0], frame, cum)[1]
            xm, hm = ens.mean(axis=0), hx.mean(axis=0)
            dx, dh = ens - xm, hx - hm
            pxh = dx.T @ dh / (n - 1)
            phh = dh.T @ dh / (n - 1) + np.diag(sig**2)
            kg = pxh @ np.linalg.inv(phh)
            yp = y[None, :] + sig[None, :] * self.rng.standard_normal((n, len(y)))
            ens = ens + (yp - hx) @ kg.T
        mean = ens.mean(axis=0)
        ens = mean + c.inflation * (ens - mean)
        ens = self._project(ens)
        self.ens = ens
        self.x = ens.mean(axis=0)
        self.P = np.cov(ens.T)

    # ------------------------------------------------------------------ outputs
    def status(self) -> dict[str, float]:
        sd = np.sqrt(np.maximum(np.diag(self.P), 0.0))
        n_al0 = self.k0[lf.KI["n0"]]
        return {"t": self.t, "T_C": float(self.x[I_T] - KELVIN_OFFSET), "T_sd": float(sd[I_T]), "Tw_C": float(self.x[I_TW] - KELVIN_OFFSET),
                "al_left_frac": float(self.x[I_F]), "al_left_mol": float(self.x[I_F] * n_al0), "al_left_sd_frac": float(sd[I_F]),
                "h2_mol": float(self.x[I_G]), "k_mult": float(math.exp(self.x[I_LK])), "k_mult_sd_ln": float(sd[I_LK]),
                "k_mult_best": float(self.best_k[0]) if self.best_k else float(math.exp(self.x[I_LK])),
                "R_mem_mult": float(math.exp(self.x[I_LR])), "R_mem_sd_ln": float(sd[I_LR])}

    def _samples(self, n: int) -> np.ndarray:
        if self.cfg.method == "enkf":
            idx = self.rng.integers(0, len(self.ens), n)
            return self.ens[idx].copy()
        p = 0.5 * (self.P + self.P.T) + 1e-12 * np.eye(NS)  # magic: jitter
        return self.rng.multivariate_normal(self.x, p, n)

    def forecast(self, horizon_s: float = 600.0, step_s: float = 10.0, n: int = 80) -> Forecast:
        """Rolling forecast with uncertainty: propagate samples of the filter distribution (parameters frozen)."""
        xs = self._samples(n)
        steps = int(horizon_s / step_s)
        t = self.t + step_s * np.arange(1, steps + 1)
        T = np.zeros((n, steps))
        G = np.zeros((n, steps))
        for i, x in enumerate(xs):
            for k in range(steps):
                x = self._advance(x, step_s)
                T[i, k], G[i, k] = x[I_T] - KELVIN_OFFSET, x[I_G]
        limit = self.cfg.limit_T_C
        exceed = (T > limit).any(axis=1)
        ttl = {}
        for nm, q in (("p05", 0.05), ("p50", 0.5), ("p95", 0.95)):
            first = np.array([t[np.argmax(r > limit)] - self.t if r.any() else math.inf for r in (T > limit)])
            ttl[nm] = float(np.quantile(first, q))
        return Forecast(t, np.quantile(T, 0.025, 0), np.median(T, 0), np.quantile(T, 0.975, 0), np.quantile(G, 0.025, 0),
                        np.median(G, 0), np.quantile(G, 0.975, 0), float(exceed.mean()), ttl)

    def warnings(self, horizon_s: float = 600.0) -> list[Warning_]:
        """Early warnings: forecast temperature limit, runaway-like heating, Al exhaustion, membrane resistance rise."""
        out: list[Warning_] = []
        fc = self.forecast(horizon_s)
        st = self.status()
        if fc.p_exceed_T > 0.5:  # magic: probability thresholds
            out.append(Warning_("danger", "T_LIMIT", f"{100 * fc.p_exceed_T:.0f}% chance liquid T exceeds {self.cfg.limit_T_C:.0f} C within "
                                f"{horizon_s / 60:.0f} min (median {fc.t_to_limit_s['p50']:.0f} s)."))
        elif fc.p_exceed_T > 0.1:  # magic: probability thresholds
            out.append(Warning_("warn", "T_LIMIT", f"{100 * fc.p_exceed_T:.0f}% chance liquid T exceeds {self.cfg.limit_T_C:.0f} C within "
                                f"{horizon_s / 60:.0f} min."))
        if st["T_C"] > DB.get("T_hdpe_max") - KELVIN_OFFSET:
            out.append(Warning_("danger", "HDPE_T", "Liquid temperature above HDPE service limit."))
        if st["al_left_frac"] < 0.05 and st["t"] > 0:  # magic: nearly exhausted
            out.append(Warning_("info", "AL_LOW", "Aluminium nearly exhausted (<5 %): hydrogen supply will end."))
        if st["R_mem_mult"] > 1.5 and st["R_mem_sd_ln"] < 0.3:  # magic: significant, well-determined rise
            out.append(Warning_("warn", "MEMBRANE_R", f"Membrane resistance estimated {st['R_mem_mult']:.1f}x the reference (drying/carbonation?)."))
        return out


_ = (F, field, Any)
