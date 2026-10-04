r"""L2 transport extras: particle-class Sherwood mass transfer and the bubble/precipitate blocking model.

Mass transfer (per class, characteristic length :math:`d`):

.. math:: k_{mt}=\frac{Sh\,D_{OH}}{d},\quad Sh = Sh_{min}+0.52\,Re_\varepsilon^{0.52}Sc^{1/3},\quad
          Re_\varepsilon=\frac{\varepsilon^{1/3}d^{4/3}}{\nu},\quad \varepsilon = g\,j_g+\varepsilon_{stir}

(Armenante-Kirwan suspended-particle form; bubble-induced agitation enters through the superficial gas
velocity :math:`j_g`).  Bubble population per unit active surface (attached bubbles):

.. math:: \dot N = \frac{q_s}{V_c}\Big(1-\frac{N}{N_{site}}\Big) - f_d N - k_c N\theta,\qquad
          \dot V = q_s - f_d V,\qquad R=\Big(\frac{3V}{4\pi N}\Big)^{1/3},\ \theta=\min(\pi N R^2,\theta_{max})

with Fritz departure diameter :math:`D_d = 0.0208\,\theta_c\sqrt{\sigma/(g\Delta\rho)}` and detachment
frequency switched on smoothly once :math:`R \ge R_d`.  Precipitate coverage
:math:`\theta_p = n_{gib}v_m/(A\,\delta_{layer})`.  Blocked fraction: :math:`1-(1-\theta_b)(1-\theta_p)`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..constants import G_ACC, T_REF
from ..core.params import ParamSet
from ..core.scenario import Scenario
from ..thermo import electrolyte
from ..thermo import propdb as DB


class L2Extras:
    def __init__(self, params: ParamSet, sc: Scenario) -> None:
        self.p = params
        self.stirred = sc.stirred
        self.jg_tau = params["jg_tau"]
        self.vm = DB.get("vm_gibbsite")
        self.blocking = sc.bubble_blocking

    # --- diffusion ------------------------------------------------------------------------------
    def diffusivity(self, T: float, c: float) -> float:
        """D_OH [m2/s]: Stokes-Einstein scaling of the dilute value with temperature and viscosity."""
        mu = electrolyte.viscosity(c, T)
        mu_ref = electrolyte.water_viscosity(T_REF)
        return self.p["D_oh_25"] * (T / T_REF) * (mu_ref / mu)

    def kmt(self, T: float, c: float, d_m: np.ndarray, jg: float, model: Any) -> np.ndarray:
        rho = electrolyte.density(c, T)
        nu = electrolyte.viscosity(c, T) / rho
        dif = self.diffusivity(T, c)
        sc_n = nu / dif
        eps = G_ACC * max(jg, 0.0) + (self.p["eps_stirred"] if self.stirred else 0.0)
        re_e = eps ** (1.0 / 3.0) * d_m ** (4.0 / 3.0) / nu
        sh = self.p["sh_min"] + self.p["sh_ak_coeff"] * re_e ** self.p["sh_ak_exp"] * sc_n ** (1.0 / 3.0)
        return np.asarray(sh * dif / d_m)

    # --- bubbles ----------------------------------------------------------------------------------
    def departure_radius(self, T: float, c: float) -> float:
        sigma = electrolyte.surface_tension(c, T)
        drho = electrolyte.density(c, T)
        d = 0.0208 * self.p["bub_contact_angle"] * math.sqrt(sigma / (G_ACC * drho))  # magic: Fritz coefficient
        return 0.5 * d

    def blocked(self, N: float, V: float, n_gib: float, a_tot: float, model: Any) -> float:
        if not self.blocking:
            return 0.0
        n = max(N, 1.0e-3)  # magic: floor
        r = (3.0 * max(V, 0.0) / (4.0 * math.pi * n)) ** (1.0 / 3.0)
        th_b = min(math.pi * N * r * r, self.p["bub_theta_max"])
        th_p = 0.0
        if n_gib > 0.0 and a_tot > 0.0:
            th_p = min(n_gib * self.vm / (a_tot * self.p["precip_layer_thickness"]), self.p["precip_theta_max"])
        return 1.0 - (1.0 - th_b) * (1.0 - th_p)

    def bubble_rates(self, N: float, V: float, q_s: float, model: Any) -> tuple[float, float]:
        if not self.blocking:
            return 0.0, 0.0
        p = self.p
        n = max(N, 1.0e-3)  # magic: floor
        r = (3.0 * max(V, 0.0) / (4.0 * math.pi * n)) ** (1.0 / 3.0)
        st = model.state_for_bubbles
        r_d = self.departure_radius(st[0], st[1])
        w = 0.1 * r_d  # magic: smoothing width of the departure switch
        sw = 1.0 / (1.0 + math.exp(-min(max((r - r_d) / w, -50.0), 50.0)))  # magic: exp guard
        fd = p["bub_fd0"] * sw
        v_c = 4.0 / 3.0 * math.pi * p["bub_R_crit"] ** 3
        theta = min(math.pi * N * r * r, p["bub_theta_max"])
        dn = q_s / v_c * max(1.0 - N / p["bub_N_site"], 0.0) - fd * N - p["bub_coal_k"] * N * theta
        dv = q_s - fd * V
        return dn, dv
