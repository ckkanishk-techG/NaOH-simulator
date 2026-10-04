r"""Sensor models (measurement science, spec 20.5).

Reading model:  :math:`y_{meas}(t) = g\,(L_\tau * y_{true})(t) + b + d\,t + \varepsilon`, with first-order
response lag :math:`L_\tau` (:math:`\tau\dot y_s = y - y_s`, e.g. thermocouple lag), gain/offset calibration
curve, linear drift and noise; the reading uncertainty combines noise, quantisation (:math:`\Delta^2/12`)
and calibration uncertainty and is propagated into every fit (:meth:`Sensor.sigma`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve


@dataclass
class Sensor:
    name: str = "sensor"
    unit: str = ""
    gain: float = 1.0
    offset: float = 0.0
    drift_per_h: float = 0.0
    tau_s: float = 0.0  # response time (first order)
    noise_sd: float = 0.0
    resolution: float = 0.0
    unc_gain_rel: float = 0.0  # 1-sigma relative calibration (gain) uncertainty
    unc_offset: float = 0.0

    # --- forward ------------------------------------------------------------------------------
    def lag(self, t: np.ndarray, y: np.ndarray) -> np.ndarray:
        """First-order lag of ``y`` sampled at (possibly non-uniform) times ``t`` (exact for piecewise-linear input)."""
        if self.tau_s <= 0.0:
            return np.asarray(y, float).copy()
        t = np.asarray(t, float)
        y = np.asarray(y, float)
        dts = np.diff(t)
        if len(t) > 2 and np.allclose(dts, dts[0], rtol=1e-9):  # uniform grid: exact IIR filter (fast path)
            from scipy.signal import lfilter, lfiltic

            dt = float(dts[0])
            a = math.exp(-dt / self.tau_s)
            c = self.tau_s * (1.0 - a) / dt
            b = [1.0 - c, -a + c]
            zi = lfiltic(b, [1.0, -a], y=[y[0]], x=[y[0]])
            return np.concatenate([[y[0]], lfilter(b, [1.0, -a], y[1:], zi=zi)[0]])
        out = np.empty(len(t))
        out[0] = y[0]
        for i in range(1, len(t)):
            dt = t[i] - t[i - 1]
            a = math.exp(-dt / self.tau_s)
            slope = (y[i] - y[i - 1]) / dt if dt > 0 else 0.0
            out[i] = y[i] - slope * self.tau_s + (out[i - 1] - y[i - 1] + slope * self.tau_s) * a
        return out

    def apply(self, t: np.ndarray, truth: np.ndarray) -> np.ndarray:
        """Expected reading (no noise) for the true signal."""
        return self.gain * self.lag(t, truth) + self.offset + self.drift_per_h * np.asarray(t) / 3600.0

    def sigma(self, reading: np.ndarray | float) -> np.ndarray:
        y = np.asarray(reading, float)
        var = self.noise_sd**2 + self.resolution**2 / 12.0 + (self.unc_gain_rel * y) ** 2 + self.unc_offset**2
        return np.sqrt(var) * np.ones_like(y)

    def simulate(self, t: np.ndarray, truth: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Synthetic reading with noise and quantisation (for demos / recovery tests)."""
        y = self.apply(t, truth) + self.noise_sd * rng.standard_normal(len(t))
        if self.resolution > 0:
            y = np.round(y / self.resolution) * self.resolution
        return y

    # --- inverse (sensor-lag deconvolution) -----------------------------------------------------
    def invert(self, t: np.ndarray, reading: np.ndarray, smooth: float = 1.0) -> np.ndarray:
        """Estimate the TRUE signal from a lagged, calibrated reading (Tikhonov-regularised deconvolution).

        Solves  min_x ||L x - r||^2 + smooth * lam ||D2 x||^2  with L the lag operator matrix, so that
        models are compared with what the true temperature was, not the delayed reading."""
        r = (np.asarray(reading, float) - self.offset - self.drift_per_h * np.asarray(t) / 3600.0) / self.gain
        n = len(t)
        if self.tau_s <= 0.0:
            return r
        lmat = np.zeros((n, n))
        for j in range(n):
            e = np.zeros(n)
            e[j] = 1.0
            lmat[:, j] = Sensor(tau_s=self.tau_s).lag(np.asarray(t, float), e)
        d2 = np.diff(np.eye(n), 2, axis=0)
        lam = smooth * 1.0e-2 * np.trace(lmat.T @ lmat) / n  # magic: relative regularisation strength
        a = lmat.T @ lmat + lam * d2.T @ d2
        return solve(a, lmat.T @ r, assume_a="pos")
