"""Versioned local property database (JSON). No physics constants are hard-coded elsewhere.

Each entry carries value, unit, plausible range, uncertainty, source placeholder and a ``status``:
``prior`` (uncalibrated modelling prior), ``unverified`` (handbook-type value, check it) or
``exact``. Never treat ``prior`` / ``unverified`` entries as established fact.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "properties.json"


@dataclass(frozen=True)
class Entry:
    name: str
    value: float
    unit: str
    min: float | None
    max: float | None
    uncertainty: float | None
    category: str
    source: str
    status: str
    dist: str | None
    sd: float | None
    desc: str
    T_min_K: float | None
    T_max_K: float | None


@lru_cache(maxsize=1)
def _load(path: str = str(DB_PATH)) -> dict[str, Entry]:
    raw = json.loads(Path(path).read_text())
    return {k: Entry(name=k, **v) for k, v in raw.items()}


def entry(name: str) -> Entry:
    try:
        return _load()[name]
    except KeyError:
        raise KeyError(f"property {name!r} not in database") from None


def trange(name: str) -> tuple[float, float]:
    """Validity temperature range [K] of an entry (must be defined)."""
    e = entry(name)
    assert e.T_min_K is not None and e.T_max_K is not None, f"{name} has no temperature range"
    return e.T_min_K, e.T_max_K


def get(name: str) -> float:
    """Value (SI) of a database property."""
    return entry(name).value


def names(category: str | None = None) -> list[str]:
    return [k for k, v in _load().items() if category is None or v.category == category]


def db_hash() -> str:
    return hashlib.sha256(Path(DB_PATH).read_bytes()).hexdigest()[:16]


def table(category: str | None = None) -> list[dict[str, Any]]:
    """Rows for the assumptions table / docs."""
    return [
        {"name": e.name, "value": e.value, "unit": e.unit, "min": e.min, "max": e.max,
         "uncertainty": e.uncertainty, "status": e.status, "source": e.source, "desc": e.desc}
        for e in _load().values() if category is None or e.category == category
    ]


def reload(path: Path | None = None) -> None:
    _load.cache_clear()
    if path is not None:
        _load(str(path))
