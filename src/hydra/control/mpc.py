r"""Model-predictive control by Dynamic Matrix Control (step-response model identified from the L1 plant).

Prediction (offset-free, anchored on the measurement :math:`y_k`):

.. math:: \hat y(k+i)=y_k+\sum_{j<k}\big[s(k-j+i)-s(k-j)\big]\Delta u_j+\sum_{l=0}^{\min(i,N_u)-1}s(i-l)\,\Delta u_{k+l}

with :math:`s(m)` the unit-step response coefficients of the L1 plant about the operating point. Each step solves
:math:`\min \|W(r-\hat y)\|^2+\lambda\|\Delta u\|^2` subject to :math:`0\le u\le u_{max}` and a rate limit
(bounded least squares), then applies the first move (receding horizon).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import lsq_linear

from .pid import step_response_diff
from .plant import DosingPlant, Obs


def identify_step_response(plant: DosingPlant, u0: float, du: float = 0.3, horizon_s: float = 240.0, warmup_s: float = 120.0,
                           meas: str = "P_bar_g") -> np.ndarray:
    """Unit-step response coefficients ``s_1..s_N`` (per mL/min) of the measurement, from the L1 plant itself."""
    _, y = step_response_diff(plant, u0, du, horizon_s, warmup_s, meas)
    return y[1:] / du


class DMC:
    def __init__(self, s: np.ndarray, setpoint: float, u_max: float, dt: float, np_h: int = 30, nu: int = 6, lam: float = 0.5,
                 u_ref: float = 0.0, rate_limit: float | None = None, measurement: str = "P_bar_g", w: float = 1.0,
                 integrating: bool = True) -> None:
        self.integrating = integrating
        self.s = np.asarray(s, float)
        self.n_m = len(self.s)
        self.sp, self.u_max, self.dt = setpoint, u_max, dt
        self.np_h, self.nu, self.lam, self.rate_limit, self.meas, self.w = np_h, nu, lam, rate_limit, measurement, w
        self.du_hist: list[float] = []
        self.u_prev = u_ref
        g = np.zeros((np_h, nu))
        for i in range(1, np_h + 1):
            for l in range(min(i, nu)):
                g[i - 1, l] = self._s(i - l)
        self.G = g

    def _s(self, m: int) -> float:
        """Step-response coefficient s(m), s(0)=0, saturating at the last identified value."""
        if m <= 0:
            return 0.0
        if m <= self.n_m:
            return float(self.s[m - 1])
        if self.integrating:  # integrating plant (gas accumulation): continue with the final slope
            slope = float(self.s[-1] - self.s[-2]) if self.n_m > 1 else 0.0
            return float(self.s[-1] + (m - self.n_m) * slope)
        return float(self.s[-1])

    def reset(self, u_ref: float = 0.0) -> None:
        self.du_hist, self.u_prev = [], u_ref

    def _free(self) -> np.ndarray:
        out = np.zeros(self.np_h)
        for age, du in enumerate(reversed(self.du_hist[-self.n_m:]), start=1):
            if du == 0.0:
                continue
            for i in range(1, self.np_h + 1):
                out[i - 1] += (self._s(age + i) - self._s(age)) * du
        return out

    def update(self, t: float, obs: Obs) -> float:
        y = float(getattr(obs, self.meas))
        pred_free = y + self._free()
        ref = np.full(self.np_h, self.sp)
        d_mat = np.eye(self.nu) - np.eye(self.nu, k=-1)  # du = D v - e0 u_prev, v = absolute inputs
        e0 = np.zeros(self.nu)
        e0[0] = self.u_prev
        a = np.vstack([self.w * (self.G @ d_mat), np.sqrt(self.lam) * d_mat])
        rhs = np.concatenate([self.w * (ref - pred_free + self.G @ e0), np.sqrt(self.lam) * e0])
        lo, hi = np.zeros(self.nu), np.full(self.nu, self.u_max)
        if self.rate_limit is not None:
            lo[0] = max(lo[0], self.u_prev - self.rate_limit * self.dt)
            hi[0] = min(hi[0], self.u_prev + self.rate_limit * self.dt)
        sol = lsq_linear(a, rhs, bounds=(lo, hi), method="bvls")
        u = float(np.clip(sol.x[0], 0.0, self.u_max))
        self.du_hist.append(u - self.u_prev)
        self.u_prev = u
        return u
