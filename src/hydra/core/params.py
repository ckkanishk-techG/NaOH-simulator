"""Fit-able model parameters (with priors) backed by the property database."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping

import numpy as np

from ..thermo import propdb as P

DEFAULT_FIT = ("k25", "Ea", "n_oh", "tau_ind", "k_mt")


class ParamSet:
    """Dict-like parameter container; values default to the property DB."""

    def __init__(self, overrides: Mapping[str, float] | None = None) -> None:
        self.v: dict[str, float] = dict(overrides or {})

    def __getitem__(self, k: str) -> float:
        return self.v[k] if k in self.v else P.get(k)

    def get(self, k: str) -> float:
        return self[k]

    def with_(self, **kw: float) -> ParamSet:
        d = dict(self.v)
        d.update(kw)
        return ParamSet(d)

    def vector(self, names: Iterable[str]) -> np.ndarray:
        return np.array([self[k] for k in names], float)

    def from_vector(self, names: Iterable[str], x: Iterable[float]) -> ParamSet:
        return self.with_(**dict(zip(names, (float(a) for a in x), strict=True)))

    def as_dict(self, names: Iterable[str] | None = None) -> dict[str, float]:
        return {k: self[k] for k in (names or self.v)}


def prior_logpdf(name: str, value: float) -> float:
    """Log prior density (up to a constant) from the DB prior: 'logn' (sd of ln), 'norm' (sd) or
    uniform on [min,max] when no distribution is given. Out-of-range -> -inf."""
    e = P.entry(name)
    if (e.min is not None and value < e.min) or (e.max is not None and value > e.max):
        return -math.inf
    if e.dist == "norm" and e.sd:
        return -0.5 * ((value - e.value) / e.sd) ** 2
    if e.dist == "logn" and e.sd and value > 0:
        return -0.5 * ((math.log(value) - math.log(e.value)) / e.sd) ** 2 - math.log(value)
    return 0.0


def bounds(names: Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = [], []
    for n in names:
        e = P.entry(n)
        lo.append(e.min if e.min is not None else -np.inf)
        hi.append(e.max if e.max is not None else np.inf)
    return np.array(lo), np.array(hi)
