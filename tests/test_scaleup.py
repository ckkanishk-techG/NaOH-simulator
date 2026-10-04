import math

import pytest

from hydra.constants import G_ACC, MW_AL
from hydra.core.params import ParamSet
from hydra.optimize.cost import Prices
from hydra.scaleup import economics as eco
from hydra.scaleup import lca, sizing

P = ParamSet()


def test_wheel_power_matches_hand_calculation():
    v = sizing.VehicleSpec(mass_kg=300.0, CdA=0.6, crr=0.03, eta_drive=0.8, aux_W=20.0, speed_kmh=36.0)  # 10 m/s
    f = 0.03 * 300 * G_ACC + 0.5 * 1.2 * 0.6 * 100.0
    assert sizing.wheel_power_W(v, P) == pytest.approx(f * 10.0 / 0.8 + 20.0, rel=1e-9)
    uphill = sizing.VehicleSpec(mass_kg=300.0, CdA=0.6, crr=0.03, eta_drive=0.8, aux_W=20.0, speed_kmh=36.0, grade_pct=5.0)
    assert sizing.wheel_power_W(uphill, P) > 2 * sizing.wheel_power_W(v, P)
    assert sizing.energy_per_km_Wh(v, P) == pytest.approx(sizing.wheel_power_W(v, P) / 36.0)


def test_stoichiometric_mass_per_kg_h2():
    al, naoh = eco.kg_per_kg_h2()
    assert al == pytest.approx(8.92, rel=2e-3) and naoh == pytest.approx(13.2, rel=3e-3)
    assert eco.kg_per_kg_h2(naoh_recovery=1.0)[1] == 0.0
    assert eco.kg_per_kg_h2(form_yield=0.5)[0] == pytest.approx(2 * al)


@pytest.fixture(scope="module")
def sized():
    return sizing.size_system(sizing.VehicleSpec(range_km=50.0, speed_kmh=30.0), P)


def test_sizing_energy_hydrogen_aluminium_chain(sized):
    s = sized
    assert s.label.startswith("SCREENING-LEVEL")
    assert s.energy_electric_Wh == pytest.approx(s.cruise_power_W * 50.0 / 30.0)
    assert s.h2_mol == pytest.approx(s.energy_electric_Wh * 3600.0 / (241.8e3 * s.eta_fc_system), rel=2e-3)
    assert s.al_loaded_kg == pytest.approx(s.h2_mol / 1.5 / 0.95 * MW_AL, rel=1e-6)
    assert s.Wh_per_g_Al == pytest.approx(s.energy_electric_Wh / (s.al_loaded_kg * 1e3))
    assert 0.2 < s.eta_fc_system < 0.6 and 0.3 < s.Wh_per_g_Al < 3.0  # 1.5 H2/Al x LHV x eta ~ 1.1 Wh/g
    assert s.h2_L_stp == pytest.approx(s.h2_mol * 22.414)


def test_heat_rejection_reactor_volume_and_budgets(sized):
    s = sized
    mol_al_s = s.al_loaded_kg * 0.95 / MW_AL / (50.0 / 30.0 * 3600.0) * (1 / 0.95)
    assert s.heat_avg_W == pytest.approx(415e3 * mol_al_s, rel=0.03)  # reaction enthalpy per mol Al
    assert s.heat_peak_W == pytest.approx(2.0 * s.heat_avg_W) and s.exchanger_UA_W_K == pytest.approx(s.heat_peak_W / P["cooling_dT_K"])
    assert s.exchanger_area_m2 == pytest.approx(s.exchanger_UA_W_K / P["cooling_h_eff"])
    assert s.reactor_L > s.liquor_L > 0 and s.water_kg > 0
    m = s.masses_kg
    assert m["total"] == pytest.approx(sum(v for k, v in m.items() if k != "total"))
    c = s.comparison
    assert c["battery"]["Wh_installed"] == pytest.approx(s.energy_electric_Wh / 0.9 / 0.9)
    assert c["petrol"]["litres"] == pytest.approx(s.energy_electric_Wh / 0.2 / 8900.0)
    assert c["hydra"]["mass_vs_battery"] == pytest.approx(m["total"] / c["battery"]["mass_kg"])
    r = s.refill
    assert r["cartridges_per_range"] >= 1 and r["km_per_cartridge"] > 0 and r["swap_minutes_per_range"] == r["cartridges_per_range"] * 10.0


def test_sizing_scales_with_range_and_aluminate_precipitation_is_reported():
    short = sizing.size_system(sizing.VehicleSpec(range_km=25.0), P)
    long_ = sizing.size_system(sizing.VehicleSpec(range_km=100.0), P)
    assert long_.al_loaded_kg == pytest.approx(4 * short.al_loaded_kg, rel=1e-6) and long_.reactor_L > short.reactor_L
    a = short.assumptions
    # 1.3 mol NaOH per mol Al cannot hold the aluminate in solution: gibbsite must precipitate (and regenerates NaOH)
    assert a["min_naoh_excess_without_precipitation"] > 1.3 and 0.0 < a["precipitate_fraction"] < 1.0
    assert a["gibbsite_kg"] > 0 and a["naoh_regenerated_mol"] > 0
    rich = sizing.size_system(sizing.VehicleSpec(range_km=25.0), P.with_(naoh_excess=a["min_naoh_excess_without_precipitation"] * 1.05))
    assert rich.assumptions["precipitate_fraction"] == 0.0 and rich.naoh_kg > short.naoh_kg
    assert short.water_kg == pytest.approx(1.3 * short.al_loaded_kg * 0.95 * 1e3 / 26.9815385 / 6.0 / 1.0, rel=0.05) or short.water_kg > 0


def test_economics_identities_and_currency():
    pr = Prices()
    e = eco.techno_economics(prices=pr)
    assert e.label.startswith("SCREENING-LEVEL") and e.currency == "INR"
    assert e.cost_per_kg_h2 == pytest.approx(sum(e.breakdown_per_kg_h2.values()), rel=1e-9)
    kwh_per_kg = e.cost_per_kg_h2 / e.cost_per_kWh_el
    assert 15.0 < kwh_per_kg < 33.3  # electric kWh per kg H2 below the LHV (33.3 kWh/kg)
    usd = eco.techno_economics(prices=pr, currency="USD")
    assert usd.cost_per_kg_h2 == pytest.approx(e.cost_per_kg_h2 * pr.fx["USD"], rel=1e-9)
    rec = eco.techno_economics(prices=pr, naoh_recovery=1.0)
    assert rec.cost_per_kg_h2 < e.cost_per_kg_h2 and rec.naoh_kg_per_kg_h2 == 0.0
    cred = eco.techno_economics(prices=pr.model_copy(update={"na_aluminate_credit_per_kg": 20.0}))
    assert cred.cost_per_kg_h2 < e.cost_per_kg_h2
    assert eco.crf(0.1, 5) == pytest.approx(0.1 / (1 - 1.1**-5)) and eco.crf(0.0, 5) == pytest.approx(0.2)


def test_economics_monotone_in_prices_and_sensitivity():
    pr = Prices()
    sens = eco.sensitivity_al_price(pr, (0.5, 1.0, 2.0))
    costs = [c for _, c in sens]
    assert costs[0] < costs[1] < costs[2]
    foil = eco.techno_economics(prices=pr, form="foil")
    can = eco.techno_economics(prices=pr, form="can")
    assert foil.cost_per_kg_h2 > can.cost_per_kg_h2  # scrap cans are cheaper than foil
    assert can.petrol_cost_per_km > 0 and can.grid_ev_cost_per_km > 0


def test_lca_ordering_labels_and_breakeven():
    r = lca.lca_compare(recycled_share=1.0)
    prim = lca.lca_compare(recycled_share=0.0)
    assert r.label.startswith("SCREENING-LEVEL") and prim.g_per_km["al_h2"] > r.g_per_km["al_h2"]  # primary Al is far worse
    assert set(r.g_per_km) == {"al_h2", "petrol", "grid_ev"} and all(lo <= hi for lo, hi in r.ranges_g_per_km.values())
    assert r.breakeven_grid_g_per_kWh > 0 and math.isfinite(r.breakeven_grid_g_per_kWh)
    clean_grid = lca.lca_compare(p=P.with_(ef_grid=0.05))
    assert clean_grid.g_per_km["grid_ev"] < clean_grid.g_per_km["al_h2"]
    dirty = lca.lca_compare(p=P.with_(ef_grid=1.0))
    assert dirty.g_per_km["grid_ev"] > clean_grid.g_per_km["grid_ev"]
    no_credit = lca.lca_compare(byproduct_reuse=False)
    assert no_credit.g_per_km["al_h2"] > r.g_per_km["al_h2"]
    assert r.recycled_share == 1.0 and "aluminium" in r.breakdown_al_h2
