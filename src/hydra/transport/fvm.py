r"""Finite-volume building blocks for the axisymmetric (r, z) solvers (L3 radial = 1 axial cell, L4 = 2-D).

Cell-centred grid on a cylinder of radius R and height H. Diffusion operator for
:math:`\partial_t(\rho\phi)=\nabla\cdot(\Gamma\nabla\phi)+S` in cylindrical coordinates with symmetry at r=0 and
Robin/Neumann conditions elsewhere; face conductivities are harmonic means. Second-order in space.
Also: method-of-manufactured-solutions (MMS) driver, theta-method time stepping and Grid-Convergence-Index /
Richardson extrapolation (ASME V&V 20 style numerical-uncertainty estimate).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import splu


@dataclass
class Grid:
    R: float
    H: float
    nr: int
    nz: int

    def __post_init__(self) -> None:
        self.dr, self.dz = self.R / self.nr, self.H / self.nz
        self.r_f = np.linspace(0.0, self.R, self.nr + 1)  # radial faces
        self.r_c = 0.5 * (self.r_f[1:] + self.r_f[:-1])
        self.z_c = (np.arange(self.nz) + 0.5) * self.dz
        # ring volumes and face areas
        self.vol_r = math.pi * (self.r_f[1:] ** 2 - self.r_f[:-1] ** 2) * self.dz  # (nr,)
        self.a_r = 2.0 * math.pi * self.r_f * self.dz  # radial face area at r_f (nr+1,)
        self.a_z = math.pi * (self.r_f[1:] ** 2 - self.r_f[:-1] ** 2)  # axial face area (nr,)
        self.n = self.nr * self.nz
        self.vol = np.tile(self.vol_r, self.nz)  # flattened index = j*nr + i
        self.rr, self.zz = np.meshgrid(self.r_c, self.z_c)
        self.ri = np.tile(np.arange(self.nr), self.nz)
        self.zj = np.repeat(np.arange(self.nz), self.nr)

    def idx(self, i: int, j: int) -> int:
        return j * self.nr + i


def harmonic(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return 2.0 * a * b / (a + b + 1e-300)  # magic: floor


def diffusion_operator(g: Grid, gamma: np.ndarray, h_wall: float = 0.0, phi_inf: float = 0.0,
                       h_bottom: float = 0.0, h_top: float = 0.0, phi_bottom: float = 0.0, phi_top: float = 0.0
                       ) -> tuple[sp.csr_matrix, np.ndarray]:
    """Return (A, b) with ``d(rho phi V)/dt = A phi + b`` for the diffusion operator (integrated over cell volumes).

    ``gamma`` is the cell-centred diffusivity/conductivity (flattened). Robin conditions
    ``-gamma dphi/dn = h (phi - phi_inf)`` at r=R (``h_wall``), z=0 (``h_bottom``) and z=H (``h_top``); r=0 symmetric."""
    n, nr, nz = g.n, g.nr, g.nz
    rows, cols, vals = [], [], []
    bvec = np.zeros(n)
    gm = gamma.reshape(nz, nr)

    def add(i0: int, j0: int, i1: int, j1: int, c: float) -> None:
        a, b = g.idx(i0, j0), g.idx(i1, j1)
        rows.extend([a, a, ])
        cols.extend([a, b])
        vals.extend([-c, c])

    for j in range(nz):
        for i in range(nr):
            if i + 1 < nr:  # radial face between i and i+1
                k = harmonic(gm[j, i], gm[j, i + 1])
                add(i, j, i + 1, j, k * g.a_r[i + 1] / g.dr)
                add(i + 1, j, i, j, k * g.a_r[i + 1] / g.dr)
            if j + 1 < nz:  # axial face between j and j+1
                k = harmonic(gm[j, i], gm[j + 1, i])
                add(i, j, i, j + 1, k * g.a_z[i] / g.dz)
                add(i, j + 1, i, j, k * g.a_z[i] / g.dz)
    diag_extra = np.zeros(n)
    if h_wall > 0.0 or math.isinf(h_wall):
        for j in range(nz):
            k = gm[j, nr - 1]
            ueff = 1.0 / (1.0 / h_wall + 0.5 * g.dr / k) if h_wall > 0 else 0.0
            a = g.idx(nr - 1, j)
            diag_extra[a] -= ueff * g.a_r[nr]
            bvec[a] += ueff * g.a_r[nr] * phi_inf
    for hh, ph, jj in ((h_bottom, phi_bottom, 0), (h_top, phi_top, nz - 1)):
        if hh > 0.0:
            for i in range(nr):
                k = gm[jj, i]
                ueff = 1.0 / (1.0 / hh + 0.5 * g.dz / k)
                a = g.idx(i, jj)
                diag_extra[a] -= ueff * g.a_z[i]
                bvec[a] += ueff * g.a_z[i] * ph
    a_mat = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    a_mat = a_mat + sp.diags(diag_extra)
    return a_mat, bvec


def theta_solve(a_mat: sp.csr_matrix, b: np.ndarray, cap: np.ndarray, phi0: np.ndarray, t_end: float, nsteps: int,
                theta: float = 0.5, source: Callable[[float], np.ndarray] | None = None) -> np.ndarray:
    """Theta-method time integration of ``cap dphi/dt = A phi + b + s(t)`` (theta=1: backward Euler, 0.5: Crank-Nicolson)."""
    dt = t_end / nsteps
    n = len(phi0)
    m = sp.diags(cap)
    lhs = (m - theta * dt * a_mat).tocsc()
    lu = splu(lhs)
    phi = phi0.copy()
    t = 0.0
    for _ in range(nsteps):
        s0 = source(t) if source else 0.0
        s1 = source(t + dt) if source else 0.0
        rhs = m @ phi + (1 - theta) * dt * (a_mat @ phi + b + s0) + theta * dt * (b + s1)
        phi = lu.solve(rhs)
        t += dt
    _ = n
    return phi


# ---------------------------------------------------------------- MMS verification
def mms_spatial_order(nrs: tuple[int, ...] = (8, 16, 32, 64), nz: int = 1, theta: float = 0.5) -> dict[str, np.ndarray]:
    """Method of manufactured solutions for the radial diffusion operator (nz = 1) or 2-D operator.

    Manufactured solution ``phi = 1 + cos(pi r / 2R) (1 + t)`` (Neumann-compatible at r=0); the source is derived
    analytically; Dirichlet-like Robin wall (large h) is used. Returns L2 errors and observed orders."""
    import sympy as s

    r, t = s.symbols("r t")
    rr = 1.0
    gamma0 = 0.7
    phi = 1 + s.cos(s.pi * r / (2 * rr)) * (1 + t)
    source = s.diff(phi, t) - (gamma0 / r) * s.diff(r * s.diff(phi, r), r)
    phi_f, src_f = s.lambdify((r, t), phi, "numpy"), s.lambdify((r, t), source, "numpy")
    dphi = s.lambdify(r, s.diff(phi, r).subs(t, 1), "numpy")
    errs = []
    hs = []
    t_end = 0.5
    for nr in nrs:
        g = Grid(rr, 1.0, nr, nz)
        gam = np.full(g.n, gamma0)
        # exact Robin wall: -gamma dphi/dr = h (phi - phi_inf) with phi_inf chosen to reproduce the manufactured flux
        h_w = 1.0e6  # magic: near-Dirichlet; phi_inf follows the exact boundary value
        a_mat, _ = diffusion_operator(g, gam, h_wall=h_w, phi_inf=0.0)
        cap = g.vol.copy()

        def b_t(tt: float, g: Grid = g) -> np.ndarray:
            _ = tt
            return np.zeros(g.n)

        def src(tt: float, g: Grid = g, h_w: float = h_w, gam: np.ndarray = gam) -> np.ndarray:
            s_vec = src_f(g.rr.ravel(), tt) * g.vol
            wall = np.zeros(g.n)
            for j in range(g.nz):
                a = g.idx(g.nr - 1, j)
                ueff = 1.0 / (1.0 / h_w + 0.5 * g.dr / gam[a])
                wall[a] += ueff * g.a_r[g.nr] * phi_f(rr, tt)
            return s_vec + wall

        phi0 = phi_f(g.rr.ravel(), 0.0)
        out = theta_solve(a_mat, np.zeros(g.n), cap, phi0, t_end, nsteps=int(40 * nr), theta=theta, source=src)  # magic: dt refined with the grid
        exact = phi_f(g.rr.ravel(), t_end)
        errs.append(float(np.sqrt(np.sum(g.vol * (out - exact) ** 2) / np.sum(g.vol))))
        hs.append(g.dr)
    errs_a, hs_a = np.array(errs), np.array(hs)
    orders = np.log(errs_a[:-1] / errs_a[1:]) / np.log(hs_a[:-1] / hs_a[1:])
    _ = dphi
    return {"h": hs_a, "err": errs_a, "order": orders}


def theta_temporal_order(theta: float, dts: tuple[int, ...] = (10, 20, 40, 80)) -> np.ndarray:
    """Observed temporal order of the theta scheme on a smooth linear problem against the matrix-exponential solution."""
    from scipy.linalg import expm

    g = Grid(1.0, 1.0, 24, 1)
    a_mat, b = diffusion_operator(g, np.full(g.n, 0.5), h_wall=2.0, phi_inf=0.0)
    phi0 = np.cos(math.pi * g.r_c / 2.0)
    cap = g.vol
    a_dense = np.diag(1.0 / cap) @ a_mat.toarray()
    exact = expm(a_dense * 0.3) @ phi0  # magic: t_end
    errs = [float(np.linalg.norm(theta_solve(a_mat, b, cap, phi0, 0.3, n, theta) - exact)) for n in dts]  # magic: t_end
    return np.log(np.array(errs[:-1]) / np.array(errs[1:])) / math.log(2.0)


# ---------------------------------------------------------------- Richardson / GCI
def gci(f_coarse: float, f_medium: float, f_fine: float, r: float = 2.0, fs: float = 1.25) -> dict[str, float]:
    """Grid Convergence Index (Roache/ASME V&V 20) from three solutions with refinement ratio ``r``.

    Returns observed order p, Richardson-extrapolated value and the GCI (relative numerical uncertainty of the
    fine-grid value, safety factor ``fs`` = 1.25 for three-grid studies)."""
    e21, e32 = f_medium - f_coarse, f_fine - f_medium
    if abs(e32) < 1e-300 or abs(e21) < 1e-300:  # magic: floor
        return {"p": float("nan"), "f_extrapolated": f_fine, "gci_fine": 0.0, "monotone": 1.0}
    ratio = e21 / e32
    if ratio <= 0:  # oscillatory convergence: fall back to the largest observed change
        return {"p": float("nan"), "f_extrapolated": f_fine, "gci_fine": fs * abs(e32 / (f_fine + 1e-300)), "monotone": 0.0}  # magic: floor
    p = math.log(ratio) / math.log(r)
    fx = f_fine + (f_fine - f_medium) / (r**p - 1.0)
    gci_f = fs * abs((f_fine - f_medium) / (f_fine + 1e-300)) / (r**p - 1.0)  # magic: floor
    return {"p": float(p), "f_extrapolated": float(fx), "gci_fine": float(gci_f), "monotone": 1.0}
