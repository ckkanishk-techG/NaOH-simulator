r"""Hardware-in-the-loop mode: READ-ONLY by default.

Any actuation output needs (1) ``actuation_enabled=True`` in the config, (2) an explicit acknowledgement that a
physical emergency stop and a pressure-relief device exist INDEPENDENTLY of this software, and (3) a software
interlock tied to the safety limits. Without all three, :class:`HILRunner` raises :class:`ActuationRefused` before
anything can be written to hardware. The software interlock is a convenience, never a substitute for hardware
protection.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel

from .plant import Obs
from .sim import Controller, Interlock

HARDWARE_WARNING = (
    "HARDWARE SAFETY: a physical emergency stop and a mechanical pressure-relief device MUST exist and work "
    "independently of this software. The software interlock only adds a second, non-sufficient layer. Do not "
    "enable actuation without supervision, ventilation and protective equipment."
)


class ActuationRefused(RuntimeError):
    pass


class HILConfig(BaseModel):
    actuation_enabled: bool = False
    ack_independent_estop_and_relief: bool = False
    read_only: bool = True
    max_u_mL_min: float = 5.0
    watchdog_s: float = 10.0  # command is forced to zero if no fresh sensor data within this time


class ActuatorSink:
    """Where commands go. The default sink only logs (read-only)."""

    def write(self, u_mL_min: float) -> None:
        raise NotImplementedError


class NullSink(ActuatorSink):
    def __init__(self) -> None:
        self.log: list[float] = []

    def write(self, u_mL_min: float) -> None:  # never touches hardware
        self.log.append(u_mL_min)


@dataclass
class HILRunner:
    controller: Controller
    config: HILConfig = field(default_factory=HILConfig)
    interlock: Interlock | None = None
    sink: ActuatorSink = field(default_factory=NullSink)
    clock: Callable[[], float] = time.monotonic
    last_data: float = field(default_factory=time.monotonic)
    commands: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        wants_actuation = self.config.actuation_enabled or not self.config.read_only or not isinstance(self.sink, NullSink)
        if wants_actuation:
            if not self.config.actuation_enabled:
                raise ActuationRefused("a real actuator sink requires actuation_enabled=True")
            if not self.config.ack_independent_estop_and_relief:
                raise ActuationRefused("set ack_independent_estop_and_relief=True only if a physical E-stop and a "
                                       "pressure-relief device exist independently of the software. " + HARDWARE_WARNING)
            if self.interlock is None:
                raise ActuationRefused("actuation requires a software Interlock tied to the safety limits")
            self.config.read_only = False

    def step(self, obs: Obs) -> dict[str, Any]:
        """Compute the proposed command; write it ONLY if actuation is enabled; always interlock-filter."""
        now = self.clock()
        self.last_data = now
        u = float(min(max(self.controller.update(obs.t, obs), 0.0), self.config.max_u_mL_min))
        filtered = self.interlock.filter(u, obs) if self.interlock else u
        written = False
        if not self.config.read_only:
            self.sink.write(filtered)
            written = True
        self.commands.append(filtered)
        return {"proposed": u, "filtered": filtered, "written": written, "read_only": self.config.read_only}

    def watchdog(self) -> bool:
        """True (and a zero command written, if actuating) when sensor data is stale."""
        stale = self.clock() - self.last_data > self.config.watchdog_s
        if stale and not self.config.read_only:
            self.sink.write(0.0)
        return stale
