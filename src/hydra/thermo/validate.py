"""Validate property models against tabulated literature data that YOU enter (spec 20.2).

CSV columns: ``property,T_K,c_mol_L,value[,unc]`` where ``property`` is one of
``density, viscosity, conductivity, surface_tension, water_activity, solubility_Al_mol_kg``.
Errors are reported per property across the T and molarity range covered by your data; no data ships
with HYDRA, so nothing here has been validated until you load a table.
"""

from __future__ import annotations

import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import electrolyte, pitzer, solubility

MODELS = {
    "density": lambda T, c: electrolyte.density(c, T),
    "viscosity": lambda T, c: electrolyte.viscosity(c, T),
    "conductivity": lambda T, c: electrolyte.conductivity(c, T),
    "surface_tension": lambda T, c: electrolyte.surface_tension(c, T),
    "water_activity": lambda T, c: pitzer.water_activity(electrolyte.molality_from_molarity(c, T), 0.0, T),
    "solubility_Al_mol_kg": lambda T, c: solubility.equilibrium_aluminate(electrolyte.molality_from_molarity(c, T), T),
}


def load_table(path: Path) -> list[dict[str, Any]]:
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows.append({"property": r["property"], "T": float(r["T_K"]), "c": float(r["c_mol_L"]),
                         "value": float(r["value"]), "unc": float(r["unc"]) if r.get("unc") else None})
    return rows


def validate(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Per-property error statistics (relative): n, rmse, max_abs, bias, T and c ranges."""
    by: dict[str, list[tuple[float, float, float, float]]] = defaultdict(list)
    for r in rows:
        pred = MODELS[r["property"]](r["T"], r["c"])
        by[r["property"]].append((r["T"], r["c"], (pred - r["value"]) / r["value"], pred))
    out = {}
    for k, v in by.items():
        e = [x[2] for x in v]
        out[k] = {"n": len(e), "rmse_rel": math.sqrt(sum(x * x for x in e) / len(e)),
                  "max_abs_rel": max(abs(x) for x in e), "bias_rel": sum(e) / len(e),
                  "T_min": min(x[0] for x in v), "T_max": max(x[0] for x in v),
                  "c_min": min(x[1] for x in v), "c_max": max(x[1] for x in v)}
    return out
