"""Scenario (input) models. Pydantic everywhere; no physics here."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

Form = Literal["foil", "powder", "wire", "can", "flake"]
Alloy = Literal["pure", "1xxx", "3xxx", "6061"]
Mode = Literal["sealed", "relief", "open", "regulated", "demand"]


class PSDSpec(BaseModel):
    """Particle size distribution over the characteristic dimension (um)."""

    kind: Literal["mono", "lognormal", "rosin_rammler", "custom"] = "mono"
    d50_um: float | None = None  # defaults to Scenario.dim_um
    sigma_g: float = Field(1.5, ge=1.0)  # lognormal geometric std
    n_rr: float = Field(2.0, gt=0)  # Rosin-Rammler spread
    n_bins: int = Field(20, ge=1, le=400)
    custom_d_um: list[float] | None = None  # custom: bin sizes
    custom_mass_frac: list[float] | None = None  # custom: mass fractions


class Scenario(BaseModel):
    # --- aluminium ---
    al_mass_g: float = Field(5.0, gt=0)
    form: Form = "foil"
    dim_um: float = Field(20.0, gt=0)  # foil/flake/can thickness, sphere/wire diameter
    piece_mm: float = Field(20.0, gt=0)  # sheet edge length / wire length
    alloy: Alloy = "pure"
    lacquer_removed: float = Field(1.0, ge=0, le=1)  # 1 = fully stripped can
    activator_ppm: float = Field(0.0, ge=0)
    psd: PSDSpec | None = None
    # --- electrolyte ---
    c_naoh_M: float = Field(2.0, gt=0)
    v_liq_mL: float = Field(200.0, gt=0)
    T0_C: float = 25.0
    T_amb_C: float = 25.0
    stirred: bool = False
    # --- vessel / gas handling ---
    v_vessel_mL: float = Field(500.0, gt=0)
    mode: Mode = "relief"
    p_relief_bar_g: float = Field(1.5, gt=0)
    vent_diameter_mm: float = Field(3.0, gt=0)  # 'open' mode orifice
    eos: Literal["ideal", "abel_noble", "peng_robinson"] = "ideal"
    cooling_UA_W_K: float = Field(0.0, ge=0)
    coolant_T_C: float = 15.0
    forced_convection: bool = False
    adiabatic: bool = False  # no heat loss to ambient (wall is a thermal mass only)
    # --- load / demand ---
    cells: int = Field(4, ge=0)
    cell_area_cm2: float = Field(25.0, gt=0)
    load_steps: list[tuple[float, float]] = Field(
        default_factory=lambda: [(0.0, 0.0), (120.0, 1.0), (900.0, 3.0), (2000.0, 1.5), (2800.0, 0.0)]
    )  # (t_s, cell current A), step-hold
    demand_h2_mol_s: list[tuple[float, float]] = Field(default_factory=list)  # 'demand' mode
    # --- room (safety) ---
    room_volume_m3: float = Field(30.0, gt=0)
    room_ach: float = Field(0.5, ge=0)
    # --- run ---
    duration_s: float = Field(3600.0, gt=0)
    rate_model: Literal["arrhenius", "mass_transfer", "electrochemical"] = "arrhenius"
    precipitation: bool = False
    polymorph: Literal["gibbsite", "bayerite"] = "gibbsite"
    activity_model: Literal["pitzer", "debye_huckel"] = "pitzer"
    fidelity: Literal["L1", "L2"] = "L1"
    bubble_blocking: bool = False  # L2: surface blocking by bubbles/precipitate (unconstrained params -> off)

    @field_validator("load_steps", "demand_h2_mol_s")
    @classmethod
    def _sorted_times(cls, v: list[tuple[float, float]]) -> list[tuple[float, float]]:
        ts = [t for t, _ in v]
        if ts != sorted(ts):
            raise ValueError("profile times must be non-decreasing")
        return v

    @model_validator(mode="after")
    def _volume_check(self) -> Scenario:
        if self.v_liq_mL >= self.v_vessel_mL:
            raise ValueError("liquid volume must be smaller than vessel volume")
        return self


def benchmark_scenario(**kw: object) -> Scenario:
    """The stage-benchmark: 5 g Al foil, 200 mL 2 M NaOH, 25 C, sealed 500 mL HDPE vessel with a
    1.5 bar(g) relief valve feeding a 4-cell 5x5 cm AEM stack under a step load."""
    base: dict[str, object] = {"mode": "regulated"}
    base.update(kw)
    return Scenario(**base)  # type: ignore[arg-type]
