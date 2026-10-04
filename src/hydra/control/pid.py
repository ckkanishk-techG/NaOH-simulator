r"""Discrete PID control with anti-windup, relay and step-response auto-tuning.

Positional form with derivative-on-measurement (first-order filter) and back-calculation anti-windup,

.. math:: u = K_p\big(e + \tfrac1{T_i}\!\int e\,dt - T_d\,\dot y_f\big),\qquad e = r - y,

output clamped to [0, u_max] with an optional rate limit. Tuning: Astrom-Hagglund relay feedback (ultimate gain
:math:`K_u=4d/(\pi a)`, period :math:`P_u`) with Ziegler-Nichols / Tyreus-Luyben rules, or a first-order-plus-
dead-time fit of an open-loop step response with IMC (lambda) rules.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import least_squares

from .plant import DosingPlant, Obs


@dataclass
class PIDGains:
    kp: float
    ti: float  # integral time [s] (inf = no integral)
    td: float = 0.0
    n: float = 10.0  # derivative filter factor (T_f = T_d / n)


class PID:
    """Pressure controller: positive error (pressure below set-point) increases the dosing rate."""

    def __init__(self, gains: PIDGains, setpoint: float, u_max: float, dt: float, u_bias: float = 0.0,
                 rate_limit: float | None = None, measurement: str = "P_bar_g") -> None:
        self.g, self.sp, self.u_max, self.dt, self.bias = gains, setpoint, u_max, dt, u_bias
        self.rate_limit, self.meas = rate_limit, measurement
        self.i_term = 0.0
        self.y_prev: float | None = None
        self.d_f = 0.0
        self.u_prev = u_bias

    def reset(self) -> None:
        self.i_term, self.y_prev, self.d_f, self.u_prev = 0.0, None, 0.0, self.bias

    def update(self, t: float, obs: Obs) -> float:
        y = float(getattr(obs, self.meas))
        e = self.sp - y
        g = self.g
        if self.y_prev is None:
            self.y_prev = y
        tf = g.td / g.n if g.td > 0 else 0.0
        a = tf / (tf + self.dt) if tf > 0 else 0.0
        dy = (y - self.y_prev) / self.dt
        self.d_f = a * self.d_f + (1.0 - a) * dy
        self.y_prev = y
        p = g.kp * e
        u_unsat = self.bias + p + self.i_term - g.kp * g.td * self.d_f
        u = float(np.clip(u_unsat, 0.0, self.u_max))
        if self.rate_limit is not None:
            u = float(np.clip(u, self.u_prev - self.rate_limit * self.dt, self.u_prev + self.rate_limit * self.dt))
        # back-calculation anti-windup (tracking time = sqrt(Ti Td) or Ti/2)
        if math.isfinite(g.ti):
            tt = math.sqrt(g.ti * g.td) if g.td > 0 else 0.5 * g.ti
            self.i_term += self.dt * (g.kp / g.ti * e + (u - u_unsat) / tt)
        self.u_prev = u
        return u


# ---------------------------------------------------------------- tuning rules
def zn_gains(ku: float, pu: float, rule: str = "tyreus_luyben") -> PIDGains:
    if rule == "ziegler_nichols":
        return PIDGains(0.6 * ku, 0.5 * pu, 0.125 * pu)
    if rule == "pi_zn":
        return PIDGains(0.45 * ku, pu / 1.2)
    return PIDGains(0.3125 * ku, 2.2 * pu)  # Tyreus-Luyben PI-like (conservative, no derivative)


def imc_gains(k: float, tau: float, theta: float, lam: float | None = None) -> PIDGains:
    """IMC (lambda) PI rule for FOPDT ``k exp(-theta s)/(tau s + 1)``; lambda defaults to max(theta, 0.25 tau)."""
    lam = lam if lam is not None else max(theta, 0.25 * tau)
    return PIDGains(tau / (k * (lam + theta)), tau)


def fopdt_fit(t: np.ndarray, y: np.ndarray, du: float) -> tuple[float, float, float]:
    """Least-squares FOPDT fit (K, tau, theta) of a step response y(t) (deviation from the initial value)."""
    y = y - y[0]

    def model(x: np.ndarray) -> np.ndarray:
        k, tau, th = x
        resp = np.where(t > th, k * du * (1.0 - np.exp(-(t - th) / max(tau, 1e-6))), 0.0)  # magic: floor
        return resp

    k0 = y[-1] / du if du != 0 else 1.0
    best = None
    for tau0 in (0.1 * t[-1], 0.3 * t[-1]):
        s = least_squares(lambda x: model(x) - y, [k0, tau0, 0.05 * t[-1]], bounds=([-np.inf, 1e-3, 0.0], [np.inf, 10 * t[-1], 0.5 * t[-1]]))  # magic: bounds
        if best is None or s.cost < best.cost:
            best = s
    assert best is not None
    return float(best.x[0]), float(best.x[1]), float(best.x[2])


def step_test(plant: DosingPlant, u0: float, du: float, horizon_s: float, warmup_s: float = 120.0, meas: str = "P_bar_g"
              ) -> tuple[np.ndarray, np.ndarray]:
    """Open-loop step response about the operating point reached with ``u0`` (state is restored afterwards)."""
    snap = plant.snapshot()
    for _ in range(int(warmup_s / plant.dt)):
        plant.step(u0)
    base = plant.snapshot()
    n = int(horizon_s / plant.dt)
    ts, ys = [0.0], [float(getattr(plant.measure(), meas))]
    for k in range(n):
        o = plant.step(u0 + du)
        ts.append((k + 1) * plant.dt)
        ys.append(float(getattr(o, meas)))
    plant.restore(base)
    plant.restore(snap)
    return np.array(ts), np.array(ys)


def step_response_diff(plant: DosingPlant, u0: float, du: float, horizon_s: float, warmup_s: float = 120.0,
                       meas: str = "P_bar_g") -> tuple[np.ndarray, np.ndarray]:
    """Step response as the DIFFERENCE between a run with ``u0 + du`` and a baseline run with ``u0`` from the same
    state (removes any drift of the baseline, e.g. gas accumulation of an integrating plant)."""
    snap = plant.snapshot()
    for _ in range(int(warmup_s / plant.dt)):
        plant.step(u0)
    base = plant.snapshot()
    n = int(horizon_s / plant.dt)
    y_base, y_step = [float(getattr(plant.measure(), meas))], [float(getattr(plant.measure(), meas))]
    for _ in range(n):
        y_base.append(float(getattr(plant.step(u0), meas)))
    plant.restore(base)
    for _ in range(n):
        y_step.append(float(getattr(plant.step(u0 + du), meas)))
    plant.restore(snap)
    return np.arange(n + 1) * plant.dt, np.array(y_step) - np.array(y_base)


def autotune_step(plant: DosingPlant, u0: float, du: float = 0.5, horizon_s: float = 240.0) -> tuple[PIDGains, dict[str, Any]]:
    """Step-response FOPDT identification with IMC PI tuning (for self-regulating responses)."""
    t, y = step_response_diff(plant, u0, du, horizon_s)
    k, tau, th = fopdt_fit(t, y, du)
    return imc_gains(k, tau, th), {"K": k, "tau": tau, "theta": th, "t": t, "y": y}


def autotune_relay(plant: DosingPlant, setpoint: float, u_mid: float, d: float, duration_s: float = 600.0, hyst: float = 0.005,
                   rule: str = "tyreus_luyben") -> tuple[PIDGains, dict[str, Any]]:
    """Astrom-Hagglund relay feedback test: u = u_mid +- d switching on the pressure error with hysteresis."""
    snap = plant.snapshot()
    u = u_mid + d
    ys, ts, sw = [], [], []
    for _ in range(int(duration_s / plant.dt)):
        o = plant.step(u)
        y = o.P_bar_g
        ys.append(y)
        ts.append(o.t)
        if u > u_mid and y > setpoint + hyst:
            u = u_mid - d
            sw.append(o.t)
        elif u < u_mid and y < setpoint - hyst:
            u = u_mid + d
            sw.append(o.t)
    plant.restore(snap)
    if len(sw) < 5:  # magic: need a few relay cycles
        raise RuntimeError("relay test did not oscillate: increase d or duration")
    y_arr, t_arr = np.array(ys), np.array(ts)
    last = t_arr >= sw[2]  # discard the first cycles
    a = 0.5 * (y_arr[last].max() - y_arr[last].min())
    pu = float(np.mean(np.diff(sw[2:]))) * 2.0
    ku = 4.0 * d / (math.pi * a)
    return zn_gains(ku, pu, rule), {"ku": ku, "pu": pu, "amplitude": a, "switch_times": sw, "t": t_arr, "y": y_arr}
