"""Damkoehler regime map: kinetic vs mass-transfer control over (T, c) for a given particle size."""

from __future__ import annotations

import numpy as np

from ..core.params import ParamSet
from ..core.rates import ArrheniusModel, damkohler
from ..core.scenario import Scenario
from ..thermo import propdb as DB
from .l2 import L2Extras

_REGIME_LO = 0.1  # magic: Da below this -> kinetic control (standard rule of thumb)
_REGIME_HI = 10.0  # magic: Da above this -> mass-transfer control


def regime_map(sc: Scenario, params: ParamSet | None = None, jg: float = 0.0,
               temps: np.ndarray | None = None, concs: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Da(T, c) on a grid with regime labels 0=kinetic, 1=mixed, 2=mass-transfer controlled."""
    p = params or ParamSet()
    temps = np.linspace(288.15, 363.15, 16) if temps is None else temps  # magic: default grid 15-90 C
    concs = np.linspace(0.25, 6.0, 20) if concs is None else concs
    ex = L2Extras(p, sc)
    d = np.array([sc.dim_um * 1.0e-6])
    model = ArrheniusModel(p, 1.0)
    da = np.zeros((len(temps), len(concs)))
    for i, T in enumerate(temps):
        for j, c in enumerate(concs):
            kmt = float(ex.kmt(float(T), float(c), d, jg, None)[0])
            da[i, j] = damkohler(model.kr(float(T)), model.n, float(c), kmt)
    lab = np.where(da < _REGIME_LO, 0, np.where(da > _REGIME_HI, 2, 1))
    _ = DB
    return {"T": temps, "c": concs, "Da": da, "regime": lab}
