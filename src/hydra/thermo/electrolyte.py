"""NaOH(aq) bulk properties: density, viscosity, conductivity, surface tension, Cp.

All correlations are *placeholders fitted to nothing yet*: validate them against your tabulated
data with ``hydra.thermo.validate`` and refit. Each correlation carries its own valid range;
outside it the range guard fires.
"""

from __future__ import annotations

from ..constants import MW_NAOH, T_REF
from . import propdb as P
from . import species
from .ranges import check

_RHO_W = P.get("rho_water_25")
_RHO_C = P.get("rho_slope_c")
_RHO_BT = P.get("rho_beta_T")
_RHO_T = P.trange("rho_slope_c")
_C_MAX = 6.0  # mol/L validity of the linear-in-c density correlation  # magic:
_T_REF = T_REF


def density(c_mol_l: float, T: float = 298.15) -> float:
    """Solution density [kg m-3] for NaOH concentration ``c`` [mol/L]."""
    check("electrolyte.density", "T", T, *_RHO_T)
    check("electrolyte.density", "c", c_mol_l, 0.0, _C_MAX)
    return (_RHO_W + _RHO_C * c_mol_l) * (1.0 - _RHO_BT * (T - _T_REF))


def water_viscosity(T: float) -> float:
    """Pure-water viscosity [Pa s] (Vogel form, base-10)."""
    return P.get("visc_water_A") * 10.0 ** (P.get("visc_water_B") / (T - P.get("visc_water_C")))


def viscosity(c_mol_l: float, T: float = 298.15) -> float:
    """Solution dynamic viscosity [Pa s] (Jones-Dole-like in c)."""
    check("electrolyte.viscosity", "T", T, *_RHO_T)
    check("electrolyte.viscosity", "c", c_mol_l, 0.0, _C_MAX)
    return water_viscosity(T) * (1.0 + P.get("visc_c1") * c_mol_l + P.get("visc_c2") * c_mol_l**2)


def conductivity(c_mol_l: float, T: float = 298.15) -> float:
    """Ionic conductivity [S m-1]."""
    check("electrolyte.conductivity", "T", T, *_RHO_T)
    check("electrolyte.conductivity", "c", c_mol_l, 0.0, _C_MAX)
    lam = P.get("cond_lambda0") / (1.0 + P.get("cond_k") * c_mol_l)
    return c_mol_l * 1.0e3 * lam * (1.0 + P.get("cond_alpha_T") * (T - _T_REF))


def surface_tension(c_mol_l: float, T: float = 298.15) -> float:
    """Surface tension [N m-1]."""
    check("electrolyte.surface_tension", "T", T, *_RHO_T)
    return (P.get("sigma_water_25") + P.get("sigma_slope_T") * (T - _T_REF)
            + P.get("sigma_slope_c") * c_mol_l)


def molality_from_molarity(c_mol_l: float, T: float = 298.15) -> float:
    """mol NaOH per kg water from mol/L (neglects dissolved aluminate)."""
    rho = density(c_mol_l, T)
    kg_water_per_l = rho * 1.0e-3 - c_mol_l * MW_NAOH
    return c_mol_l / max(kg_water_per_l, 1.0e-9)


def solution_cp_per_litre(c_mol_l: float) -> float:
    """Heat capacity of 1 L of NaOH(aq) [J K-1] from the species heat capacities."""
    rho = density(c_mol_l)
    kg_w = rho * 1.0e-3 - c_mol_l * MW_NAOH
    from ..constants import MW_H2O
    return kg_w / MW_H2O * species.cp("H2O(l)") + c_mol_l * (species.cp("Na+(aq)") + species.cp("OH-(aq)"))


def density_from_mass(n_na: float, m_sol: float, T: float) -> tuple[float, float]:
    """Density [kg m-3] and Na concentration [mol/L] for ``n_na`` mol Na+ in ``m_sol`` kg solution.

    Closed-form solution of rho = (a + b c) with c = n rho / (1000 m)."""
    k = 1.0 - _RHO_BT * (T - _T_REF)
    a, b = k * _RHO_W, k * _RHO_C
    rho = a / (1.0 - b * n_na / (1.0e3 * m_sol))
    c = n_na * rho / (1.0e3 * m_sol)
    check("electrolyte.density", "c", c, 0.0, _C_MAX)
    check("electrolyte.density", "T", T, *_RHO_T)
    return rho, c
