"""Sensor sources for the live twin: serial (Arduino/ESP32), Modbus (injected client), streaming CSV, simulated.

Frame format (CSV line, 1 Hz, from the supplied firmware):  ``t_s,T1_C,T2_C,P_bar_g,flow_L_min,I_A,V_V``
Missing channels are ``nan``. Everything here is READ-ONLY: there is no actuator code in this module.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..core import l1_fast as lf
from ..core.params import ParamSet
from ..core.scenario import Scenario
from ..inference.sensors import Sensor

FIELDS = ("t_s", "T1_C", "T2_C", "P_bar_g", "flow_L_min", "I_A", "V_V")
NAN = float("nan")


@dataclass
class Frame:
    t: float
    T1: float = NAN  # liquid temperature [C]
    T2: float = NAN  # wall / ambient temperature [C]
    P: float = NAN  # gauge pressure [bar]
    flow: float = NAN  # H2 flow at STP [L/min]
    I: float = NAN  # stack current [A]
    V: float = NAN  # stack voltage [V]


def parse_line(line: str) -> Frame | None:
    """Parse one CSV line; returns None for headers/comments/garbage (never raises on bad data)."""
    s = line.strip()
    if not s or s.startswith(("#", "t_s", "t_ms")):
        return None
    parts = s.split(",")
    if len(parts) < 2:
        return None
    try:
        vals = [float(p) if p.strip().lower() not in ("", "nan", "na", "null") else NAN for p in parts[: len(FIELDS)]]
    except ValueError:
        return None
    vals += [NAN] * (len(FIELDS) - len(vals))
    if not math.isfinite(vals[0]):
        return None
    return Frame(*vals)


def format_line(f: Frame) -> str:
    return ",".join("nan" if not math.isfinite(v) else f"{v:.6g}" for v in (f.t, f.T1, f.T2, f.P, f.flow, f.I, f.V))


class SerialSource:
    """Read frames from a serial port (requires ``pyserial``; port opened read-only in practice - nothing is written)."""

    def __init__(self, port: str, baud: int = 115200, timeout: float = 2.0, opener: Callable[..., Any] | None = None) -> None:
        if opener is None:
            import serial  # pyserial

            opener = serial.Serial
        self.ser = opener(port, baud, timeout=timeout)

    def __iter__(self) -> Iterator[Frame]:
        while True:
            raw = self.ser.readline()
            if not raw:
                return
            f = parse_line(raw.decode(errors="ignore"))
            if f is not None:
                yield f


class CsvStreamSource:
    """Follow a growing CSV file (``tail -f`` style); stops after ``idle_s`` seconds without new data."""

    def __init__(self, path: str | Path, idle_s: float = 5.0, poll_s: float = 0.2, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        self.path, self.idle_s, self.poll_s, self.clock, self.sleep = Path(path), idle_s, poll_s, clock, sleep

    def __iter__(self) -> Iterator[Frame]:
        last = self.clock()
        with open(self.path) as fh:
            while True:
                line = fh.readline()
                if line:
                    last = self.clock()
                    f = parse_line(line)
                    if f is not None:
                        yield f
                elif self.clock() - last > self.idle_s:
                    return
                else:
                    self.sleep(self.poll_s)


class ModbusSource:
    """Poll holding registers through an injected client (e.g. ``pymodbus``'s ``ModbusTcpClient``).

    ``register_map``: channel -> (address, scale, offset); values are ``raw*scale+offset``. Read-only: only
    ``read_holding_registers`` is ever called."""

    def __init__(self, client: Any, register_map: dict[str, tuple[int, float, float]], unit: int = 1,
                 clock: Callable[[], float] = time.monotonic) -> None:
        bad = set(register_map) - {"T1", "T2", "P", "flow", "I", "V"}
        if bad:
            raise ValueError(f"unknown channels {bad}")
        self.client, self.map, self.unit, self.clock = client, register_map, unit, clock
        self.t0 = clock()

    def read(self) -> Frame:
        kw: dict[str, float] = {}
        for ch, (addr, scale, off) in self.map.items():
            rr = self.client.read_holding_registers(addr, 1, slave=self.unit)
            if getattr(rr, "isError", lambda: False)():
                kw[ch] = NAN
            else:
                kw[ch] = float(rr.registers[0]) * scale + off
        return Frame(self.clock() - self.t0, **kw)


class SimulatedSource:
    """Realistic demo/test data: the reduced model plays the true system; sensors add lag, calibration error,
    quantisation, noise and drift. ``hidden`` overrides the true parameters (e.g. a different rate constant)."""

    def __init__(self, sc: Scenario, hidden: dict[str, float] | None = None, dt: float = 5.0, seed: int = 0,
                 t_sensor: Sensor | None = None, wall_sensor: Sensor | None = None, sd_flow: float = 0.01,
                 r_mem_mult: float = 1.0, stack_current: float = 0.0, n_cells: int = 4, flow_sensor: bool = True) -> None:
        self.sc, self.dt = sc, dt
        self.rng = np.random.default_rng(seed)
        p = ParamSet(hidden or {})
        sim = lf.simulate_fast(lf.build_constants(sc, p, dt=dt), sc, dt)
        self.truth = sim
        self.t_sensor = t_sensor or Sensor("tc", "C", tau_s=8.0, noise_sd=0.1, resolution=0.0625, drift_per_h=0.02)
        self.w_sensor = wall_sensor or Sensor("wall", "C", tau_s=3.0, noise_sd=0.1)
        self.read_T = self.t_sensor.apply(sim["t"], sim["T"] - 273.15)
        self.read_W = self.w_sensor.apply(sim["t"], sim["Tw"] - 273.15)
        self.sd_flow, self.r_mem_mult, self.i_stack, self.n_cells = sd_flow, r_mem_mult, stack_current, n_cells
        self.flow_sensor = flow_sensor

    def __iter__(self) -> Iterator[Frame]:
        t = self.truth["t"]
        gen = self.truth["gen"]
        from ..fuelcell.stack import Stack, StackSpec

        stack = Stack(StackSpec(n_series=self.n_cells, seed=0), ParamSet({"fc_asr": ParamSet()["fc_asr"] * self.r_mem_mult}))
        for k in range(1, len(t)):
            flow = (gen[k] - gen[k - 1]) / self.dt * 22.414 * 60.0 if self.flow_sensor else NAN  # L/min STP
            v = stack.voltage(self.i_stack) if self.i_stack > 0 else NAN
            yield Frame(float(t[k]), float(self.read_T[k] + self.t_sensor.noise_sd * self.rng.standard_normal()),
                        float(self.read_W[k] + self.w_sensor.noise_sd * self.rng.standard_normal()), 0.0,
                        float(flow + self.sd_flow * self.rng.standard_normal()) if self.flow_sensor else NAN,
                        self.i_stack if self.i_stack > 0 else NAN, float(v + 0.002 * self.rng.standard_normal()) if self.i_stack > 0 else NAN)
