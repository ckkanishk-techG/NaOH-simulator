"""Pydantic request/response models for the API (no physics here)."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from ..core.scenario import Scenario


class SimulateRequest(BaseModel):
    scenario: Scenario
    overrides: dict[str, float] = Field(default_factory=dict)
    rtol: float = Field(1e-7, gt=0, le=1e-2)
    dt_out: float = Field(1.0, gt=0)


class ChannelIn(BaseModel):
    kind: Literal["n_h2", "T", "Tw"]
    t: list[float]
    y: list[float]
    sigma: list[float] | float = 0.0
    tau_s: float = 0.0  # thermocouple lag for T channels


class ExperimentIn(BaseModel):
    id: str
    scenario: Scenario
    channels: list[ChannelIn]
    batch: str = "A"
    split: Literal["calib", "val", "blind"] = "calib"


class CalibrateRequest(BaseModel):
    experiments: list[ExperimentIn] = Field(default_factory=list)
    eln_experiments: list[str] = Field(default_factory=list)  # ids of logged experiments (sensor CSV attachments)
    parameters: list[str] = Field(default_factory=lambda: ["k25", "Ea", "n_oh", "tau_ind"])
    model: Literal["empirical", "mass_transfer", "echem"] = "empirical"
    method: Literal["lsq", "bayes"] = "lsq"
    n_steps: int = Field(300, ge=50, le=5000)
    dt: float = Field(2.0, gt=0)
    record_in_eln: bool = True


class OEDRequest(BaseModel):
    calibration: CalibrateRequest
    n_candidates: int = Field(60, ge=5, le=500)
    top_k: int = Field(3, ge=1, le=20)
    constraints: dict[str, Any] = Field(default_factory=dict)
    seed: int = 0


class OptimizeRequest(BaseModel):
    goal: dict[str, Any] = Field(default_factory=dict)
    method: Literal["de", "bo", "pareto", "robust"] = "de"
    objectives: list[str] = Field(default_factory=lambda: ["cost", "peak_T"])
    maxiter: int = Field(25, ge=2, le=300)
    seed: int = 0
    verify: bool = True


class ControlRequest(BaseModel):
    load_steps: list[tuple[float, float]] = Field(default_factory=lambda: [(0.0, 1.0), (400.0, 2.5), (900.0, 1.0), (1400.0, 3.0), (1900.0, 0.5)])
    duration_s: float = Field(2400.0, gt=60.0, le=20000.0)
    setpoint_bar_g: float = Field(0.3, gt=0.05, le=1.4)
    controllers: list[Literal["open_loop", "pid", "dmc"]] = Field(default=["open_loop", "pid", "dmc"])
    dt: float = Field(5.0, ge=1.0, le=20.0)


class SpatialRequest(BaseModel):
    scenario: Scenario
    level: Literal["L3", "L4"] = "L3"
    nr: int = Field(12, ge=4, le=80)
    nz: int = Field(8, ge=2, le=60)
    settled_fraction: float = Field(1.0, ge=0, le=1)
    gci: bool = False


class ScaleupRequest(BaseModel):
    range_km: float = Field(50.0, gt=0)
    speed_kmh: float = Field(30.0, gt=0)
    grade_pct: float = 0.0
    form: Literal["foil", "powder", "wire", "can", "flake"] = "can"
    currency: str = "INR"
    recycled_share: float = Field(1.0, ge=0, le=1)
    naoh_recovery: float = Field(0.0, ge=0, le=1)
    prices: dict[str, float] = Field(default_factory=dict)
    overrides: dict[str, float] = Field(default_factory=dict)


class ElnCreate(BaseModel):
    title: str
    operator: str = "local"
    setup: dict[str, Any] = Field(default_factory=dict)
    quantities: dict[str, Any] = Field(default_factory=dict)
    conditions: dict[str, Any] = Field(default_factory=dict)
    notes: str = ""
    batch: str = "A"
    split: Literal["calib", "val", "blind"] = "calib"


class LoginRequest(BaseModel):
    username: str
    password: str


class UserCreate(BaseModel):
    username: str
    password: str
    role: Literal["viewer", "researcher", "admin"] = "researcher"
