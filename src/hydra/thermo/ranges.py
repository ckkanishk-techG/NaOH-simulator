"""Valid-range guard: property models never extrapolate silently.

Policy ``warn`` (default) records a ``RangeViolation`` in the active collector and emits a
``RangeWarning`` once per message; ``error`` raises; ``ignore`` does nothing.
"""

from __future__ import annotations

import threading
import warnings
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass


class RangeWarning(UserWarning):
    pass


class RangeError(ValueError):
    pass


@dataclass(frozen=True)
class RangeViolation:
    model: str
    variable: str
    value: float
    lo: float
    hi: float

    def __str__(self) -> str:
        return f"{self.model}: {self.variable}={self.value:.4g} outside valid [{self.lo:.4g}, {self.hi:.4g}]"


_state = threading.local()


def _stack() -> list[dict[str, RangeViolation]]:
    if not hasattr(_state, "stack"):
        _state.stack = []
        _state.policy = "warn"
    return _state.stack  # type: ignore[no-any-return]


@contextmanager
def collect(policy: str = "warn") -> Iterator[dict[str, RangeViolation]]:
    """Collect violations (one per model+variable, worst value kept)."""
    st = _stack()
    box: dict[str, RangeViolation] = {}
    old = getattr(_state, "policy", "warn")
    st.append(box)
    _state.policy = policy
    try:
        yield box
    finally:
        st.pop()
        _state.policy = old


def check(model: str, variable: str, value: float, lo: float, hi: float) -> None:
    """Register a violation if ``value`` is outside ``[lo, hi]`` (small tolerance)."""
    tol = 1.0e-9 * max(abs(lo), abs(hi), 1.0)
    if lo - tol <= value <= hi + tol:
        return
    st = _stack()
    policy = getattr(_state, "policy", "warn")
    v = RangeViolation(model, variable, float(value), float(lo), float(hi))
    if policy == "error":
        raise RangeError(str(v))
    if policy == "ignore":
        return
    key = f"{model}:{variable}"
    if st:
        old = st[-1].get(key)
        if old is None or abs(value - (lo + hi) / 2) > abs(old.value - (lo + hi) / 2):
            st[-1][key] = v
    else:
        warnings.warn(str(v), RangeWarning, stacklevel=3)
