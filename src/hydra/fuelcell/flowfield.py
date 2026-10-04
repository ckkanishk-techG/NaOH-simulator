r"""Flow-field / nickel-foam distribution network (laminar-plus-inertial resistor network).

Channels (width w, depth h, length L, optionally filled with nickel foam) are fed by an inlet header and
drained by an outlet header, each a chain of segments with resistance :math:`k_h r_{ch}`.  Channel
resistance: Poiseuille for open channels, Darcy-Forchheimer for foam,

.. math:: \frac{\Delta P}{L}=\frac{\mu u}{K}+\rho C_F u^2,\qquad
          K=\frac{\varepsilon^3 d_p^2}{180(1-\varepsilon)^2}\ (\text{Kozeny-Carman}).

The nodal network is solved by Picard iteration on the inertial term; outputs: per-channel flow
fractions, total pressure drop, and uniformity index :math:`U = 1-(q_{max}-q_{min})/(2\bar q)`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.params import ParamSet


def foam_permeability(porosity: float, pore_d: float) -> float:
    """Kozeny-Carman permeability [m2] of the nickel foam."""
    return porosity**3 * pore_d**2 / (180.0 * (1.0 - porosity) ** 2)  # magic: Kozeny-Carman constant


@dataclass
class FlowResult:
    q_frac: np.ndarray  # fraction of total flow in each channel
    dp: float  # total pressure drop [Pa]
    uniformity: float


def channel_resistance(p: ParamSet, with_foam: bool, mu: float, q: float, rho: float) -> float:
    """Pressure-drop per unit flow [Pa s m-3] of one channel at volumetric flow ``q`` [m3/s]."""
    w, h, ln = p["ff_ch_width"], p["ff_ch_depth"], p["ff_ch_length"]
    a = w * h
    if with_foam:
        k = foam_permeability(p["foam_porosity"], p["foam_pore_d"])
        u = q / a
        return (mu / k * ln + rho * p["foam_forchheimer"] * u * ln) / a
    dh = 2.0 * w * h / (w + h)
    return 32.0 * mu * ln / (a * dh * dh)  # magic: laminar friction constant (Poiseuille, 32 mu L u / dh^2)


def solve_network(p: ParamSet, q_total: float, n: int | None = None, with_foam: bool = True,
                  mu: float | None = None, rho: float | None = None) -> FlowResult:
    n = n or int(p["ff_n_channels"])
    mu = mu or p["mu_h2"]
    rho = rho or p["rho_h2_stp"]
    q = np.full(n, q_total / n)
    for _ in range(50):  # magic: Picard iterations
        rc = np.array([channel_resistance(p, with_foam, mu, max(float(qi), 1e-15), rho) for qi in q])  # magic: floor
        rh = p["ff_header_k"] * rc.mean()
        # unknowns: inlet header pressures P_k (k=0..n-1) and outlet header Q_k; outlet at Q_{n-1}=0 (gauge)
        a = np.zeros((2 * n, 2 * n))
        b = np.zeros(2 * n)
        for k in range(n):
            # inlet node k: inflow from k-1 (or source at k=0) = channel + flow to k+1
            if k > 0:
                a[k, k - 1] += 1.0 / rh
                a[k, k] -= 1.0 / rh
            else:
                b[k] -= q_total
            if k < n - 1:
                a[k, k] -= 1.0 / rh
                a[k, k + 1] += 1.0 / rh
            a[k, k] -= 1.0 / rc[k]
            a[k, n + k] += 1.0 / rc[k]
            # outlet node k: channel inflow + flow from k-1 header = flow to k+1 (or outlet at last node)
            row = n + k
            a[row, k] += 1.0 / rc[k]
            a[row, row] -= 1.0 / rc[k]
            if k > 0:
                a[row, row - 1] += 1.0 / rh
                a[row, row] -= 1.0 / rh
            if k < n - 1:
                a[row, row] -= 1.0 / rh
                a[row, row + 1] += 1.0 / rh
            else:
                a[row, row] -= 1.0 / (rh * 1e-3)  # magic: outlet to ambient through a very small resistance
        x = np.linalg.solve(a, b)
        pin, pout = x[:n], x[n:]
        q_new = (pin - pout) / rc
        if np.max(np.abs(q_new - q)) < 1e-9 * q_total:  # magic: tolerance
            q = q_new
            break
        q = 0.5 * q + 0.5 * q_new
    q = np.maximum(q, 0.0)
    qf = q / q.sum()
    mean = qf.mean()
    uni = 1.0 - (qf.max() - qf.min()) / (2.0 * mean)
    return FlowResult(q_frac=qf, dp=float(pin[0] - pout[-1]), uniformity=float(uni))


def sheet_flow_factors(p: ParamSet, n_cells: int) -> np.ndarray:
    """Relative reactant flow reaching each cell of a stack fed through a common manifold (mean = 1)."""
    res = solve_network(p, q_total=1.0e-6, n=n_cells, with_foam=False)  # magic: reference flow, laminar regime
    return res.q_frac * n_cells


def foam_pressure_drop(p: ParamSet, face_velocity: float) -> float:
    """Pressure drop [Pa] across the foam thickness at a given face velocity [m/s]."""
    k = foam_permeability(p["foam_porosity"], p["foam_pore_d"])
    t = p["foam_thickness"]
    return t * (p["mu_h2"] / k * face_velocity + p["rho_h2_stp"] * p["foam_forchheimer"] * face_velocity**2)

