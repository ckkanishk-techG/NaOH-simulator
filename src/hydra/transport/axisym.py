r"""L3 (1-D radial) and L4 (2-D axisymmetric) spatial models of the reacting liquid + settled particle bed.

Per cell: temperature :math:`T`, hydroxide and aluminate moles, remaining particle volume fraction :math:`f`,
film activity :math:`\psi`; plus one wall node. With ``nz = 1`` and no axial heat loss this is the 1-D radial
(L3) model; ``nz > 1`` adds the axial direction, the settled bed (solids fraction, reduced liquid porosity,
Bruggeman effective diffusivity, Maxwell effective conductivity) and bottom heat loss (L4).

.. math:: \epsilon\partial_t c=\nabla\!\cdot(D\epsilon^{1.5}Nu\,\nabla c) - \dot r/V,\qquad
          C\partial_t T=\nabla\!\cdot(k_{eff}Nu\,\nabla T)-\Delta H(T)\,\dot r/V

Natural convection is approximated by an effective-conductivity/diffusivity enhancement
:math:`Nu=\max(1,c\,Ra^{1/3})` (``convection=True``) or a user multiplier ``conv_mult`` (use a huge value to
recover the well-mixed limit: L3 -> L2 and L4 -> L3 consistency tests). Reaction kinetics are the empirical
Arrhenius law with oxide film; all spatial discretisation is second order (MMS-verified, see ``fvm``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp
from scipy.integrate import solve_ivp

from ..constants import G_ACC, KELVIN_OFFSET, MW_H2O, SIGMA_SB, T_REF, R
from ..core.geometry import area_fraction, make_bins
from ..core.params import ParamSet
from ..core.scenario import Scenario
from ..thermo import electrolyte, species
from ..thermo import propdb as DB
from . import fvm

N_OUT_DEFAULT = 120  # magic: default number of output times


@dataclass
class SpatialResult:
    t: np.ndarray
    series: dict[str, np.ndarray]
    fields: list[dict[str, np.ndarray]]  # snapshots: t, T, c, f (each shape (nz, nr))
    grid: fvm.Grid
    summary: dict[str, float]
    numerical_uncertainty: dict[str, dict[str, float]] = field(default_factory=dict)


def _diffusivity(T: float, c: float, p: ParamSet) -> float:
    mu = electrolyte.viscosity(c, T)
    return p["D_oh_25"] * (T / T_REF) * (electrolyte.water_viscosity(T_REF) / mu)


def maxwell_conductivity(k_l: float, k_s: float, phi: np.ndarray) -> np.ndarray:
    """Maxwell-Garnett effective conductivity of dispersed solids (volume fraction phi)."""
    beta = (k_s - k_l) / (k_s + 2.0 * k_l)
    return k_l * (1.0 + 2.0 * beta * phi) / (1.0 - beta * phi)


class AxisymModel:
    def __init__(self, sc: Scenario, params: ParamSet | None = None, nr: int = 20, nz: int = 1, settled_fraction: float = 0.0,
                 convection: bool = True, conv_mult: float | None = None, bottom_loss: bool = False,
                 wall_loss: bool = True) -> None:
        self.sc, self.p = sc, params or ParamSet()
        p = self.p
        self.nr, self.nz = nr, nz
        bins = make_bins(sc.model_copy(update={"psd": None}))
        self.n0_tot, self.a0_tot, self.gexp = float(bins.n0.sum()), float(bins.a0.sum()), bins.g
        self.v_p = float(bins.v0.sum())
        V_liq = sc.v_liq_mL * 1e-6
        self.R = p["r_ves"]
        self.H = (V_liq + self.v_p) / (math.pi * self.R**2)
        g = fvm.Grid(self.R, self.H, nr, nz)
        self.g = g
        # solids distribution (settled bed + dispersed remainder)
        phi_pack = p["bed_packing"]
        v_cyl = math.pi * self.R**2 * self.H
        s = float(np.clip(settled_fraction, 0.0, 1.0))
        z_lo, z_hi = np.arange(nz) * g.dz, (np.arange(nz) + 1) * g.dz
        h_bed = s * self.v_p / (phi_pack * math.pi * self.R**2)
        if h_bed > self.H:
            raise ValueError("settled bed taller than the liquid: reduce Al mass or packing")
        overlap = np.clip(np.minimum(z_hi, h_bed) - z_lo, 0.0, None) / g.dz  # fraction of each row inside the bed
        phi_row = s * phi_pack * overlap + (1.0 - s) * self.v_p / v_cyl
        self.phi = np.repeat(phi_row, nr)  # solids volume fraction per cell, flattened (row-major j*nr+i)
        self.eps = 1.0 - self.phi
        self.h_bed = h_bed
        w = self.phi * g.vol
        self.w = w / w.sum()
        self.n0_c, self.a0_c = self.n0_tot * self.w, self.a0_tot * self.w
        self.v_liq_c = self.eps * g.vol
        c0 = sc.c_naoh_M
        self.T0 = sc.T0_C + KELVIN_OFFSET
        self.M_oh0 = c0 * 1e3 * self.v_liq_c  # mol per cell
        rho = electrolyte.density(c0, self.T0)
        nw_tot = (rho * V_liq - c0 * V_liq / 1e-3 * 39.997e-3) / MW_H2O  # magic: NaOH molar mass
        self.nw0_c = nw_tot * self.v_liq_c / self.v_liq_c.sum()
        # transport operators (constant matrices; multiplied by scalar property factors each call)
        k_l = p["k_liq"]
        k_eff = maxwell_conductivity(k_l, p["k_al_solid"], self.phi)
        self.A_k, _ = fvm.diffusion_operator(g, k_eff)
        self.A_d, _ = fvm.diffusion_operator(g, self.eps ** p["bed_bruggeman"])
        self.k_eff = k_eff
        self.conv_mult = conv_mult
        self.convection = convection and conv_mult is None
        self.bottom_loss, self.wall_loss = bottom_loss, wall_loss
        # wall face areas per cell
        self.a_wall = np.zeros(g.n)
        if wall_loss:
            for j in range(nz):
                self.a_wall[g.idx(nr - 1, j)] += g.a_r[nr]
        if bottom_loss:
            for i in range(nr):
                self.a_wall[g.idx(i, 0)] += g.a_z[i]
        if nz == 1 and bottom_loss:  # 1-D radial: the single axial cell carries the full bottom area (consistent with L1 A_in)
            pass
        self.k_hd, self.t_w = p["k_hdpe"], p["t_ves"]
        r_o = self.R + self.t_w
        self.H_ves = sc.v_vessel_mL * 1e-6 / (math.pi * self.R**2)
        self.A_out = 2 * math.pi * r_o * self.H_ves + 2 * math.pi * r_o**2
        self.C_wall = p["m_ves"] * p["cp_hdpe"]
        self.T_amb = sc.T_amb_C + KELVIN_OFFSET
        self.dh298, self.dcp = species.reaction_enthalpy(T_REF), species.reaction_delta_cp()
        self.cp = {k: species.cp(k) for k in ("H2O(l)", "Na+(aq)", "OH-(aq)", "Al(OH)4-(aq)", "Al(s)")}
        self.cp_m = DB.get("cp_liq_mass")
        self.rho_l = rho
        self.f_s = float(min(max(0.01, 0.02), 0.3))  # magic: sheet ramp width (explicit-free solver tolerates the standard value)
        self.stir = p["stirred_h_mult"] if sc.stirred else 1.0
        from ..core.rates import alloy_multiplier

        self.mult = alloy_multiplier(sc.alloy, sc.activator_ppm, p)
        n = g.n
        self.sl = {"T": slice(0, n), "M": slice(n, 2 * n), "A": slice(2 * n, 3 * n), "f": slice(3 * n, 4 * n),
                   "psi": slice(4 * n, 5 * n)}
        self.i_tw, self.i_gen = 5 * n, 5 * n + 1
        self.ny = 5 * n + 2

    # ------------------------------------------------------------------ helpers
    def initial_state(self) -> np.ndarray:
        n = self.g.n
        y = np.zeros(self.ny)
        y[self.sl["T"]] = self.T0
        y[self.sl["M"]] = self.M_oh0
        y[self.sl["f"]] = 1.0
        y[self.i_tw] = self.T0
        _ = n
        return y

    def _nu_eff(self, dT: float) -> float:
        if self.conv_mult is not None:
            return float(self.conv_mult)
        if not self.convection:
            return 1.0
        p = self.p
        nu = electrolyte.viscosity(self.sc.c_naoh_M, self.T0) / self.rho_l
        al = p["k_liq"] / (self.rho_l * self.cp_m)
        ra = G_ACC * p["rho_beta_T"] * max(dT, 0.0) * self.H**3 / (nu * al)
        return max(1.0, p["nat_conv_coeff"] * ra ** (1.0 / 3.0)) if ra > p["nat_conv_ra_min"] else 1.0

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        g, p, sl = self.g, self.p, self.sl
        T, M, A, f, psi = (y[sl[k]] for k in ("T", "M", "A", "f", "psi"))
        Tw = y[self.i_tw]
        f = np.maximum(f, 0.0)
        M = np.maximum(M, 0.0)
        A = np.maximum(A, 0.0)
        c = M / (1e3 * self.v_liq_c)
        c_al = A / (1e3 * self.v_liq_c)
        # reaction
        kr = p["k25"] * self.mult * np.exp(-p["Ea"] / R * (1.0 / T - 1.0 / T_REF))
        j = kr * psi * np.maximum(c, 1e-12) ** p["n_oh"]  # magic: floor
        r = j * self.a0_c * area_fraction(f, self.gexp, self.f_s)
        r = np.where(c > 1e-9, r, 0.0)  # magic: no reaction without hydroxide
        df = np.where(self.n0_c > 0, -r / np.maximum(self.n0_c, 1e-300), 0.0)  # magic: floor
        # transport
        w_all = self.a_wall > 0
        T_b = float(np.sum(T[w_all] * self.a_wall[w_all]) / np.sum(self.a_wall[w_all])) if w_all.any() else float(T.mean())
        nu = self._nu_eff(float(T.max() - T_b))
        d_scalar = _diffusivity(float(T.mean()), max(float(c.mean()), 0.05), p) * nu  # magic: floor
        dM = (d_scalar * 1e3 * (self.A_d @ c)) - r  # moles/s: flux of (mol/m3 * m2 * m2/s / m) -> c in mol/L => *1e3
        dA = (d_scalar * 1e3 * (self.A_d @ c_al)) + r
        # energy
        dh = self.dh298 + self.dcp * (T - T_REF)
        n_w = self.nw0_c - 3.0 * (self.n0_c - f * self.n0_c)
        n_na = M + A
        cp = self.cp
        cap = n_w * cp["H2O(l)"] + n_na * cp["Na+(aq)"] + M * cp["OH-(aq)"] + A * cp["Al(OH)4-(aq)"] + f * self.n0_c * cp["Al(s)"]
        q_rx = -dh * r
        q_cond = nu * (self.A_k @ T)
        # wall coupling (natural convection inside, wall conduction)
        dT_w = abs(T_b - Tw)
        h_l = self.H
        nu_l = electrolyte.viscosity(self.sc.c_naoh_M, self.T0) / self.rho_l
        al_l = p["k_liq"] / (self.rho_l * self.cp_m)
        ra = G_ACC * p["rho_beta_T"] * max(dT_w, 0.1) * h_l**3 / (nu_l * al_l)  # magic: floor
        nu_in = (0.825 + 0.387 * ra ** (1 / 6) / (1 + (0.492 * al_l / nu_l) ** (9 / 16)) ** (8 / 27)) ** 2  # magic: Churchill-Chu
        h_in = nu_in * p["k_liq"] / h_l * self.stir
        u_in = 1.0 / (1.0 / h_in + self.t_w / (2.0 * self.k_hd))
        k_cell = self.k_eff * nu
        dist = np.where(self.ri_is_wall, 0.5 * g.dr, 0.5 * g.dz)
        u_eff = np.where(self.a_wall > 0, 1.0 / (1.0 / u_in + dist / k_cell), 0.0)
        q_w = u_eff * self.a_wall * (T - Tw)  # W lost from each cell to the wall
        dT = (q_rx + q_cond - q_w) / cap
        # wall node
        a_f = self.a_wall.sum()
        _ = a_f
        t_f = 0.5 * (Tw + self.T_amb)
        ra_o = G_ACC / t_f * max(abs(Tw - self.T_amb), 0.1) * self.H_ves**3 / (DB.get("air_nu") * DB.get("air_alpha"))  # magic: floor
        nu_o = (0.825 + 0.387 * ra_o ** (1 / 6) / (1 + (0.492 / DB.get("air_Pr")) ** (9 / 16)) ** (8 / 27)) ** 2  # magic: Churchill-Chu
        h_o = nu_o * DB.get("air_k") / self.H_ves
        q_amb = (Tw - self.T_amb) * h_o * self.A_out / (1.0 + h_o * self.t_w / (2.0 * self.k_hd))
        q_amb += p["emis_hdpe"] * SIGMA_SB * self.A_out * (Tw**4 - self.T_amb**4)
        if self.sc.adiabatic:
            q_amb = 0.0
        dy = np.empty_like(y)
        dy[sl["T"]], dy[sl["M"]], dy[sl["A"]], dy[sl["f"]] = dT, dM, dA, df
        dy[sl["psi"]] = (1.0 - psi) / p["tau_ind"]
        dy[self.i_tw] = (q_w.sum() - q_amb) / self.C_wall
        dy[self.i_gen] = 1.5 * r.sum()
        return dy

    @property
    def ri_is_wall(self) -> np.ndarray:
        """True where the wall face is radial (distance = dr/2), else axial."""
        out = np.zeros(self.g.n, bool)
        for j in range(self.nz):
            out[self.g.idx(self.nr - 1, j)] = True
        return out

    def jac_sparsity(self) -> sp.csr_matrix:
        g = self.g
        n = g.n
        lap = (abs(self.A_k) + abs(self.A_d)).tocsr()
        lap.data[:] = 1.0
        eye = sp.identity(n, format="csr")
        blocks = [[None] * 5 for _ in range(5)]
        for a in range(5):
            for b in range(5):
                blocks[a][b] = (lap + eye) if (a in (0, 1, 2) and b in (0, 1, 2)) else eye if (b in (0, 1, 3, 4) or a == b) else None
        big = sp.bmat(blocks, format="lil")
        s = sp.lil_matrix((self.ny, self.ny))
        s[:5 * n, :5 * n] = big
        s[:n, self.i_tw] = 1
        s[self.i_tw, :n] = 1
        s[self.i_tw, self.i_tw] = 1
        s[self.i_gen, :] = 1
        return s.tocsr()

    def run(self, t_end: float | None = None, n_out: int = 120, n_snap: int = 5, rtol: float = 1e-6, atol: float = 1e-9) -> SpatialResult:
        t_end = t_end or self.sc.duration_s
        y0 = self.initial_state()
        t_eval = np.linspace(0.0, t_end, n_out + 1)
        sol = solve_ivp(self.rhs, (0.0, t_end), y0, method="BDF", t_eval=t_eval, rtol=rtol, atol=atol,
                        jac_sparsity=self.jac_sparsity(), max_step=t_end / 50)
        if not sol.success:
            raise RuntimeError(sol.message)
        sl, g = self.sl, self.g
        Tm = np.array([np.sum(sol.y[sl["T"], k] * g.vol) / g.vol.sum() for k in range(sol.y.shape[1])])
        gen = sol.y[self.i_gen]
        fields = []
        for k in np.linspace(0, sol.y.shape[1] - 1, n_snap).astype(int):
            yk = sol.y[:, k]
            fields.append({"t": float(sol.t[k]), "T": yk[sl["T"]].reshape(self.nz, self.nr),
                           "c": (yk[sl["M"]] / (1e3 * self.v_liq_c)).reshape(self.nz, self.nr),
                           "f": yk[sl["f"]].reshape(self.nz, self.nr)})
        series = {"gen": gen, "T_mean": Tm, "T_max": sol.y[sl["T"]].max(axis=0), "T_min": sol.y[sl["T"]].min(axis=0),
                  "Tw": sol.y[self.i_tw], "conversion": gen / (1.5 * self.n0_tot)}
        gap = float(np.max(series["T_max"] - series["T_min"]))
        i90 = int(np.searchsorted(gen, 0.9 * gen[-1])) if gen[-1] > 0 else len(gen) - 1  # magic: 90 %
        summary = {"h2_total_mol": float(gen[-1]), "peak_T_max_K": float(series["T_max"].max()), "peak_T_mean_K": float(Tm.max()),
                   "max_T_spread_K": gap, "t90_s": float(sol.t[min(i90, len(gen) - 1)]), "bed_height_m": self.h_bed}
        return SpatialResult(sol.t, series, fields, g, summary)


def simulate_l3(sc: Scenario, params: ParamSet | None = None, nr: int = 20, **kw: Any) -> SpatialResult:
    """1-D radial model (single axial cell, no axial heat loss, optional bottom loss lumped over the radius)."""
    return AxisymModel(sc, params, nr=nr, nz=1, settled_fraction=0.0, **{k: v for k, v in kw.items() if k not in ("nz", "t_end", "n_out")}).run(
        kw.get("t_end"), kw.get("n_out", N_OUT_DEFAULT))


def simulate_l4(sc: Scenario, params: ParamSet | None = None, nr: int = 10, nz: int = 10, settled_fraction: float = 1.0,
                **kw: Any) -> SpatialResult:
    """2-D axisymmetric model with a settled particle bed and bottom heat loss."""
    kw.setdefault("bottom_loss", True)
    return AxisymModel(sc, params, nr=nr, nz=nz, settled_fraction=settled_fraction,
                       **{k: v for k, v in kw.items() if k not in ("t_end", "n_out")}).run(kw.get("t_end"), kw.get("n_out", N_OUT_DEFAULT))


def with_gci(factory: Any, levels: tuple[tuple[int, int], ...] = ((6, 1), (12, 1), (24, 1)), keys: tuple[str, ...] = (
        "h2_total_mol", "peak_T_max_K", "t90_s")) -> tuple[SpatialResult, dict[str, dict[str, float]]]:
    """Run ``factory(nr, nz)`` on three grids (refinement ratio 2) and attach the GCI numerical uncertainty."""
    results = [factory(nr, nz) for nr, nz in levels]
    unc = {k: fvm.gci(results[0].summary[k], results[1].summary[k], results[2].summary[k]) for k in keys}
    out = results[-1]
    out.numerical_uncertainty = unc
    return out, unc


_ = SpatialResult
