r"""Species thermochemistry and reaction energetics with consistency checks.

Species data (298.15 K) come from the property database; heat capacities are taken constant, so

.. math:: h_i(T) = \Delta_f H_i^\circ + c_{p,i}(T - T_{ref}),\qquad
          s_i(T) = s_i^\circ + c_{p,i}\ln(T/T_{ref}).

Reaction (ionic form, per mole Al):  Al + OH- + 3 H2O -> Al(OH)4- + 3/2 H2
"""

from __future__ import annotations

import math

from ..constants import T_REF
from . import propdb as P

# per-mole-Al stoichiometry (negative = consumed)
REACTION_AL: dict[str, float] = {
    "Al(s)": -1.0, "OH-(aq)": -1.0, "H2O(l)": -3.0, "Al(OH)4-(aq)": 1.0, "H2(g)": 1.5,
}
# aluminate precipitation: Al(OH)4- -> Al(OH)3(s) + OH-
REACTION_PRECIP: dict[str, float] = {"Al(OH)4-(aq)": -1.0, "Al(OH)3(gibbsite)": 1.0, "OH-(aq)": 1.0}

SPECIES = ("Al(s)", "H2O(l)", "H2O(g)", "OH-(aq)", "Na+(aq)", "Al(OH)4-(aq)",
           "Al(OH)3(gibbsite)", "H2(g)", "H2(aq)", "air(g)")


def cp(sp: str) -> float:
    return P.get(f"cp:{sp}")


def h(sp: str, T: float) -> float:
    """Molar enthalpy [J mol-1] relative to elements at 298.15 K (formation basis)."""
    return P.get(f"hf:{sp}") + cp(sp) * (T - T_REF)


def s(sp: str, T: float) -> float:
    return P.get(f"s0:{sp}") + cp(sp) * math.log(T / T_REF)


def g_formation_298(sp: str) -> float:
    return P.get(f"gf:{sp}")


def reaction_enthalpy(T: float, stoich: dict[str, float] = REACTION_AL) -> float:
    r"""Reaction enthalpy per mole Al [J mol-1], with Kirchhoff correction
    :math:`\Delta H(T)=\Delta H(T_{ref})+\Delta C_p (T-T_{ref})`."""
    return sum(nu * h(sp, T) for sp, nu in stoich.items())


def reaction_gibbs(T: float, stoich: dict[str, float] = REACTION_AL) -> float:
    """Standard reaction Gibbs energy per mole Al from H(T)-T S(T) [J mol-1]."""
    return sum(nu * (h(sp, T) - T * s(sp, T)) for sp, nu in stoich.items())


def reaction_delta_cp(stoich: dict[str, float] = REACTION_AL) -> float:
    return sum(nu * cp(sp) for sp, nu in stoich.items())


def gibbs_consistency_298(stoich: dict[str, float] = REACTION_AL) -> tuple[float, float, float]:
    """Hess/Gibbs-Helmholtz closure at 298.15 K.

    Returns (dG from formation Gibbs energies, dG from dH - T dS, difference) in J mol-1.
    """
    dg_f = sum(nu * g_formation_298(sp) for sp, nu in stoich.items())
    dh = sum(nu * P.get(f"hf:{sp}") for sp, nu in stoich.items())
    ds = sum(nu * P.get(f"s0:{sp}") for sp, nu in stoich.items())
    dg_hs = dh - T_REF * ds
    return dg_f, dg_hs, dg_f - dg_hs
