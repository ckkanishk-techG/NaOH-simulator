"""Cost model with user-entered local prices (default currency INR, switchable).

All prices are PLACEHOLDERS ("[PRICE-TODO]"): replace them with your local quotes. Nothing here is a quotation."""

from __future__ import annotations

from pydantic import BaseModel, Field

from ..constants import MW_NAOH
from ..core.scenario import Scenario

# currency units per INR (user-editable; placeholders, not market data)
FX_PER_INR = {"INR": 1.0, "USD": 0.012, "EUR": 0.011, "GBP": 0.0095}  # [PRICE-TODO]


class Prices(BaseModel):
    currency: str = "INR"
    al_foil_per_kg: float = 400.0  # [PRICE-TODO]
    al_powder_per_kg: float = 700.0  # [PRICE-TODO]
    al_wire_per_kg: float = 450.0  # [PRICE-TODO]
    al_flake_per_kg: float = 600.0  # [PRICE-TODO]
    al_scrap_can_per_kg: float = 150.0  # [PRICE-TODO]
    naoh_per_kg: float = 60.0  # [PRICE-TODO]
    water_per_L: float = 0.1  # [PRICE-TODO]
    cooling_per_W_K: float = 15.0  # [PRICE-TODO] per W/K of heat-exchange capacity
    cell_cost: float = 800.0  # [PRICE-TODO] per 5x5 cm AEM cell
    battery_per_Wh: float = 40.0  # [PRICE-TODO]
    vessel_per_L: float = 80.0  # [PRICE-TODO]
    na_aluminate_credit_per_kg: float = 0.0  # [PRICE-TODO] by-product credit (0 = none)
    grid_per_kWh: float = 8.0  # [PRICE-TODO]
    petrol_per_L: float = 100.0  # [PRICE-TODO]
    fx: dict[str, float] = Field(default_factory=lambda: dict(FX_PER_INR))

    def convert(self, inr: float, to: str | None = None) -> float:
        """Convert an INR amount to ``to`` (default: ``currency``) using the user-set rates."""
        return inr * self.fx[to or self.currency]


def al_price(form: str, p: Prices) -> float:
    return {"foil": p.al_foil_per_kg, "powder": p.al_powder_per_kg, "wire": p.al_wire_per_kg, "flake": p.al_flake_per_kg,
            "can": p.al_scrap_can_per_kg}[form]


def consumables_cost(sc: Scenario, p: Prices) -> float:
    """Cost of Al + NaOH + water for one charge [INR]."""
    naoh_kg = sc.c_naoh_M * sc.v_liq_mL / 1e3 * MW_NAOH
    return (sc.al_mass_g / 1e3 * al_price(sc.form, p) + naoh_kg * p.naoh_per_kg + sc.v_liq_mL / 1e3 * p.water_per_L)


def hardware_cost(sc: Scenario, p: Prices, n_cells: int = 0, battery_Wh: float = 0.0) -> float:
    return (sc.v_vessel_mL / 1e3 * p.vessel_per_L + sc.cooling_UA_W_K * p.cooling_per_W_K + n_cells * p.cell_cost
            + battery_Wh * p.battery_per_Wh)


def total_cost(sc: Scenario, p: Prices, n_cells: int = 0, battery_Wh: float = 0.0) -> float:
    return consumables_cost(sc, p) + hardware_cost(sc, p, n_cells, battery_Wh)
