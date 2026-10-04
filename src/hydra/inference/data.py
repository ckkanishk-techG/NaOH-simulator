"""Experiment/dataset containers, synthetic-data generators and parameter transforms."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..core import l1_fast as lf
from ..core.params import ParamSet, bounds
from ..core.scenario import Scenario
from ..thermo import propdb as DB
from .corrections import Reading, moles_with_uncertainty
from .sensors import Sensor

KINDS = ("n_h2", "T", "Tw")  # cumulative H2 moles, liquid temperature reading, wall temperature reading


@dataclass
class Channel:
    kind: str  # one of KINDS
    t: np.ndarray
    y: np.ndarray
    sigma: np.ndarray  # 1-sigma of each reading (propagated sensor/correction uncertainty)
    sensor: Sensor | None = None  # temperature channels: forward model of the reading (lag, gain, offset, drift)

    def __post_init__(self) -> None:
        assert self.kind in KINDS, self.kind
        self.t, self.y, self.sigma = (np.asarray(a, float) for a in (self.t, self.y, self.sigma))


@dataclass
class Experiment:
    id: str
    scenario: Scenario
    channels: list[Channel]
    batch: str = "A"
    split: str = "calib"  # 'calib' | 'val' | 'blind'
    weight: float = 1.0
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def n_points(self) -> int:
        return int(sum(len(c.t) for c in self.channels))


def h2_channel_from_readings(t: np.ndarray, readings: list[Reading], mc: int = 400) -> Channel:
    """Convert raw gas-volume (or mass-loss) readings into a cumulative-moles channel with propagated sigma."""
    n, s = zip(*[moles_with_uncertainty(r, mc, seed=i) for i, r in enumerate(readings)], strict=True)
    return Channel("n_h2", np.asarray(t), np.asarray(n), np.maximum(np.asarray(s), 1e-9), None)  # magic: floor


def synthetic_experiment(eid: str, sc: Scenario, truth: ParamSet, rng: np.random.Generator, model: str = "empirical",
                         sd_n: float = 2.0e-4, sd_T: float = 0.15, t_sensor: Sensor | None = None,
                         dt: float = 2.0, n_pts: int = 40, batch: str = "A", split: str = "calib",
                         kinds: tuple[str, ...] = ("n_h2", "T")) -> Experiment:
    """Synthetic data from the reduced model (labelled synthetic; used for recovery tests and demos)."""
    sim = lf.simulate_fast(lf.build_constants(sc, truth, model, dt), sc, dt)
    t_obs = np.linspace(sc.duration_s / n_pts, sc.duration_s, n_pts)
    chans = []
    if "n_h2" in kinds:
        y = np.interp(t_obs, sim["t"], sim["gen"]) + sd_n * rng.standard_normal(n_pts)
        chans.append(Channel("n_h2", t_obs, y, np.full(n_pts, sd_n)))
    if "T" in kinds:
        sens = t_sensor or Sensor("thermocouple", "K", tau_s=10.0, noise_sd=sd_T)
        read = sens.apply(sim["t"], sim["T"])
        y = np.interp(t_obs, sim["t"], read) + sd_T * rng.standard_normal(n_pts)
        chans.append(Channel("T", t_obs, y, np.full(n_pts, sd_T), sens))
    if "Tw" in kinds:
        y = np.interp(t_obs, sim["t"], sim["Tw"]) + sd_T * rng.standard_normal(n_pts)
        chans.append(Channel("Tw", t_obs, y, np.full(n_pts, sd_T), Sensor("wall", "K")))
    return Experiment(eid, sc, chans, batch, split, 1.0, {"synthetic": True, "truth": truth.v})


# ---------------------------------------------------------------- parameter space
class ParamSpace:
    """Maps named parameters <-> optimiser coordinates ``x`` (O(1), decorrelated).

    Natural coordinates ``u``: ``ln(value)`` for log-normal priors, else the value. If ``k25``, ``Ea`` and
    ``n_oh`` are all fitted, the rate constant is re-expressed at the data-centre ``(T_c, c_c)``:

    ``ln k_c = ln k25 - Ea/R (1/T_c - 1/T25) + n ln c_c``  (unit-determinant shear), which removes the
    banana-shaped Arrhenius degeneracy that stalls gradient optimisers and MCMC.  Then
    ``x = (u' - centre)/scale`` with centre = DB default and scale = prior sd (or a quarter of the plausible
    range). Bounds are the DB plausible ranges on the natural parameters."""

    def __init__(self, names: list[str], ref: tuple[float, float] | None = None) -> None:
        self.names = list(names)
        self.entries = [DB.entry(n) for n in names]
        self.is_log = np.array([e.dist == "logn" for e in self.entries])
        lo, hi = bounds(names)
        self.ulo = np.where(self.is_log, np.log(np.maximum(lo, 1e-300)), lo)  # magic: floor
        self.uhi = np.where(self.is_log, np.log(hi), hi)
        centre_u = np.array([np.log(e.value) if lg else e.value for e, lg in zip(self.entries, self.is_log, strict=True)])
        self.scale = np.array([e.sd if (e.dist in ("norm", "logn") and e.sd) else (self.uhi[k] - self.ulo[k]) / 4.0
                               for k, e in enumerate(self.entries)])
        self.shear = None
        if ref is not None and all(n in self.names for n in ("k25", "Ea", "n_oh")):
            self.shear = (self.names.index("k25"), self.names.index("Ea"), self.names.index("n_oh"), ref[0], ref[1])
        self.centre_u = centre_u
        self.centre = self._shear(centre_u)
        self.lo = (self.ulo - self.centre_u) / self.scale - 6.0  # magic: generous box; exact test via natural bounds
        self.hi = (self.uhi - self.centre_u) / self.scale + 6.0  # magic: generous box; exact test via natural bounds
        if self.shear is None:
            self.lo, self.hi = (self.ulo - self.centre) / self.scale, (self.uhi - self.centre) / self.scale

    def _shear(self, u: np.ndarray, inverse: bool = False) -> np.ndarray:
        if self.shear is None:
            return u
        ik, ie, inn, t_c, c_c = self.shear
        from ..constants import T_REF, R

        u = u.copy()
        sign = 1.0 if inverse else -1.0
        term = -u[ie] / R * (1.0 / t_c - 1.0 / T_REF) + u[inn] * np.log(c_c)
        u[ik] = u[ik] + sign * term
        return u

    def _natural(self, x: np.ndarray) -> np.ndarray:
        return self._shear(np.asarray(x) * self.scale + self.centre, inverse=True)

    def to_u(self, theta: dict[str, float] | np.ndarray) -> np.ndarray:
        v = np.array([theta[n] for n in self.names]) if isinstance(theta, dict) else np.asarray(theta, float)
        u = np.where(self.is_log, np.log(v), v)
        return (self._shear(u) - self.centre) / self.scale

    def from_u(self, x: np.ndarray) -> dict[str, float]:
        u = self._natural(x)
        v = np.where(self.is_log, np.exp(np.minimum(u, 700.0)), u)  # magic: exp overflow guard
        return {n: float(a) for n, a in zip(self.names, v, strict=True)}

    def jac_natural(self) -> np.ndarray:
        """Matrix A with du_natural = A dx (linear map of the decorrelating shear and scaling)."""
        a = np.diag(self.scale.copy())
        if self.shear is not None:
            from ..constants import T_REF, R

            ik, ie, inn, t_c, c_c = self.shear
            a[ik, ie] = self.scale[ie] / R * (1.0 / t_c - 1.0 / T_REF)
            a[ik, inn] = -self.scale[inn] * np.log(c_c)
        return a

    def in_bounds(self, x: np.ndarray) -> bool:
        u = self._natural(x)
        return bool(np.all(u >= self.ulo) and np.all(u <= self.uhi))

    def log_prior(self, x: np.ndarray) -> float:
        """Log prior density (shear has unit determinant, so evaluate the natural-coordinate prior)."""
        if not self.in_bounds(x):
            return -np.inf
        u = self._natural(x)
        lp = 0.0
        for i, e in enumerate(self.entries):
            if e.dist in ("norm", "logn") and e.sd:
                lp += -0.5 * ((u[i] - self.centre_u[i]) / self.scale[i]) ** 2
        return float(lp)

    def default_u(self) -> np.ndarray:
        return np.zeros(len(self.names))

    @classmethod
    def for_experiments(cls, names: list[str], exps: list[Experiment]) -> ParamSpace:
        t_c = float(np.mean([e.scenario.T0_C for e in exps])) + 273.15 + 2.0  # magic: mean heating of the reacting liquid
        c_c = float(np.exp(np.mean([np.log(e.scenario.c_naoh_M) for e in exps])))
        return cls(names, (t_c, c_c))
