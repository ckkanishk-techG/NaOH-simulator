import numpy as np
import pytest

from hydra.constants import F
from hydra.core.params import ParamSet
from hydra.core.scenario import Scenario, benchmark_scenario
from hydra.fuelcell import flowfield
from hydra.fuelcell.bop import DCDC, Battery, Conditioner
from hydra.fuelcell.cell import AEMCell, CellCond, bv_overpotential, e_rev
from hydra.fuelcell.stack import Stack, StackSpec
from hydra.system.model import SystemModel, SystemSpec, simulate_system
from hydra.thermo import water

P = ParamSet()


def test_reversible_voltage_matches_textbook_and_falls_with_T():
    assert e_rev(298.15, 1.0, 1.0, 1.0) == pytest.approx(1.229, abs=0.003)
    assert e_rev(333.15, 1.0, 1.0, 1.0) < e_rev(298.15, 1.0, 1.0, 1.0)
    assert e_rev(298.15, 2.0, 1.0) > e_rev(298.15, 1.0, 1.0)  # Nernst: pressure raises E


def test_butler_volmer_inversion_round_trip():
    f = F / (8.314462618 * 320.0)
    for i in (1.0, 50.0, 2000.0):
        eta = bv_overpotential(i, 5.0, 0.7, 0.4, 320.0)
        back = 5.0 * (np.exp(0.7 * f * eta) - np.exp(-0.4 * f * eta))
        assert back == pytest.approx(i, rel=1e-6)
    assert bv_overpotential(0.0, 5.0, 0.5, 0.5, 300.0) == 0.0


def test_polarization_curve_monotone_with_losses_and_limit():
    cell, c = AEMCell(), CellCond(T=323.15, lam=12.0)
    v = [cell.voltage(i, c)["V"] for i in (0, 200, 1000, 3000, 5000, 5800)]
    assert all(a > b for a, b in zip(v, v[1:]))
    d = cell.voltage(3000.0, c)
    assert d["V"] == pytest.approx(d["E"] - d["eta_a"] - d["eta_c"] - d["eta_ohm"] - d["eta_conc"])
    assert cell.voltage(5990.0, c)["V"] < 0.6 * cell.voltage(1000.0, c)["V"]  # collapse near iL
    assert cell.voltage(3000.0, c, 2.0)["E"] > cell.voltage(3000.0, c, 1.0)["E"]


def test_asr_trends_temperature_hydration_carbonate_ageing():
    cell = AEMCell()
    base = cell.asr(CellCond(T=323.15, lam=12.0))
    assert base == pytest.approx(P["fc_asr"], rel=1e-9)
    assert cell.asr(CellCond(T=343.15, lam=12.0)) < base
    assert cell.asr(CellCond(T=323.15, lam=6.0)) > base
    assert cell.asr(CellCond(T=323.15, lam=12.0, x_carb=0.5)) > base
    assert cell.asr(CellCond(T=323.15, lam=12.0, age_mem=1.2)) == pytest.approx(1.2 * base)


def test_water_management_flooding_and_drying():
    cell = AEMCell()
    c = CellCond(T=323.15, lam=12.0)
    i = 4000.0
    # dry air + dry anode: water removed faster than made at low current, flooding at saturation humidity
    wet = ParamSet({"fc_RH_air": 1.0, "fc_k_evap_c": 1e-4, "fc_k_evap_a": 1e-5})
    cw = AEMCell(wet)
    lam = 12.0
    for _ in range(4000):
        d, _ = cw.water_rates(i, CellCond(T=323.15, lam=lam), 1.0, 1.0)
        lam += 1.0 * d
    assert cw.flood_risk(lam) > 0.0 or lam > 20.0
    dlam_dry, _ = cell.water_rates(0.0, c, 0.0, 0.0)
    assert dlam_dry < 0.0  # open circuit + dry gas: membrane dries
    assert cell.dry_risk(2.0) > 0.0 and cell.dry_risk(15.0) == 0.0
    dlam_hi, _ = cell.water_rates(8000.0, c, 0.5, 0.5)
    dlam_lo, _ = cell.water_rates(100.0, c, 0.5, 0.5)
    assert dlam_hi > dlam_lo  # more current -> more product water
    assert cell.voltage(3000.0, CellCond(T=323.15, lam=24.5))["iL"] < cell.voltage(3000.0, CellCond(T=323.15, lam=12.0))["iL"]


def test_carbonation_off_by_default_and_self_purging_steady_state():
    cell = AEMCell()
    assert cell.carb_rate(1000.0, CellCond(T=323.15, lam=12.0)) == 0.0
    on = AEMCell(ParamSet({"fc_carb_on": 1.0}))
    c = CellCond(T=323.15, lam=12.0)
    assert on.carb_rate(0.0, c) > 0.0
    assert on.carb_rate(5000.0, CellCond(T=323.15, lam=12.0, x_carb=0.9)) < on.carb_rate(0.0, CellCond(T=323.15, lam=12.0, x_carb=0.9))


def test_stack_series_voltage_power_curve_and_inverse():
    s = Stack(StackSpec(n_series=4))
    v1 = s.cell_voltages(5.0)
    assert s.voltage(5.0) == pytest.approx(v1.sum())
    i_mpp, p_max = s.max_power_point()
    assert 0 < i_mpp and p_max > s.power(0.5 * i_mpp) and p_max > s.power(0.99 * s.cell.limiting_current(s.cells[0]) * s.cell.area)
    i, ok = s.current_for_power(3.0)
    assert ok and s.power(i) == pytest.approx(3.0, rel=1e-6) and i < i_mpp
    assert s.current_for_power(2 * p_max)[1] is False
    assert s.h2_flow(10.0) == pytest.approx(4 * 10.0 / (2 * F))
    sp = Stack(StackSpec(n_series=4, n_parallel=3))
    assert sp.h2_flow(30.0) == pytest.approx(12 * 10.0 / (2 * F))


def test_cell_voltage_spread_and_thermal_coupling():
    s = Stack(StackSpec(n_series=6, seed=1))
    assert s.spread(8.0)["spread_mV"] > 0.0
    th = Stack(StackSpec(n_series=6, seed=1), ParamSet({"fc_thermal_on": 1.0}))
    th.T_amb = 298.15
    for c in th.cells:
        c.T = 298.15
    for _ in range(300):
        th.step(10.0, 3.0, 0.5)
    t = np.array([c.T for c in th.cells])
    assert t.min() > 298.15 + 0.5  # heated by losses
    assert t[2:4].mean() > t[[0, -1]].mean()  # interior cells hotter than end cells


def test_ageing_off_by_default_and_effect_when_on():
    s = Stack(StackSpec(n_series=2))
    for _ in range(10):
        s.step(3600.0, 5.0, 0.5)
    assert s.cells[0].age_cat == 1.0 and s.cells[0].age_mem == 1.0
    a = Stack(StackSpec(n_series=2), ParamSet({"fc_k_cat": 0.01, "fc_k_mem": 0.02}))
    v0 = a.voltage(5.0)
    for _ in range(10):
        a.step(3600.0, 5.0, 0.5)
    assert a.voltage(5.0) < v0


def test_flow_network_properties():
    r = flowfield.solve_network(P, 1e-6)
    assert r.q_frac.sum() == pytest.approx(1.0) and 0 < r.uniformity <= 1 and r.dp > 0
    worse = flowfield.solve_network(P.with_(ff_header_k=0.5), 1e-6)
    assert worse.uniformity < r.uniformity
    one = flowfield.solve_network(P, 1e-6, n=1)
    assert one.uniformity == pytest.approx(1.0)
    assert flowfield.foam_permeability(0.95, 5e-4) > flowfield.foam_permeability(0.90, 5e-4)
    assert flowfield.foam_pressure_drop(P, 0.2) > 2 * flowfield.foam_pressure_drop(P, 0.1) - 1e-9  # inertial term
    assert flowfield.sheet_flow_factors(P, 5).mean() == pytest.approx(1.0)


def test_dcdc_inverse_and_efficiency_and_battery_limits():
    d = DCDC(P)
    for p in (0.5, 5.0, 40.0):
        assert d.p_out(d.p_in(p)) == pytest.approx(p, rel=1e-9)
    assert 0 < d.efficiency(5.0) < 1 and d.efficiency(20.0) > d.efficiency(0.5)
    b = Battery(capacity_Wh=1.0, p_max_W=10.0, soc=0.5)
    got = b.exchange(100.0, 1.0, P)
    assert got == pytest.approx(10.0)  # power limited
    for _ in range(100000):
        b.exchange(10.0, 10.0, P)
        if b.soc <= b.soc_min + 1e-9:
            break
    assert b.soc >= b.soc_min - 1e-9
    b2 = Battery(capacity_Wh=1.0, p_max_W=10.0, soc=0.9)
    e0 = b2.soc
    b2.exchange(-10.0, 3600.0, P)
    assert b2.soc <= b2.soc_max + 1e-9 and b2.soc >= e0


def test_conditioner_condensation_and_mist():
    c = Conditioner(P, condenser=True)
    hot = c.condition(363.15, 0.5, 101325.0, 1e-3, 1e-3, 323.15)
    none = Conditioner(P, condenser=False).condition(363.15, 0.5, 101325.0, 1e-3, 1e-3, 323.15)
    assert hot.water_condensed_mol_s > 0 and hot.rh_stack <= none.rh_stack
    assert hot.mist_g_s == pytest.approx(none.mist_g_s * (1 - P["cond_mist_eff"]), rel=1e-9)
    dry = Conditioner(P, condenser=True, dryer=True).condition(363.15, 0.5, 101325.0, 1e-3, 0.0, 323.15)
    assert dry.rh_stack < hot.rh_stack
    assert water.psat(323.15) > 0


@pytest.fixture(scope="module")
def bench():
    return simulate_system(benchmark_scenario())


def test_system_benchmark_conservation_and_energy_accounting(bench):
    v = bench.vessel
    assert all(abs(v.ledger[k]) < 1e-3 for k in ("Al", "Na", "O", "H", "energy"))
    assert bench.summary["Wh_stack"] >= bench.summary["Wh_load"] > 0
    assert 0 < bench.summary["stack_efficiency_energy"] < 1
    assert bench.summary["Wh_per_g_Al_loaded"] > 0 and bench.summary["system_efficiency"] < 0.5


def test_system_faraday_consistency(bench):
    dt = bench.t[1] - bench.t[0]
    n_from_current = 4 * float(np.sum(bench["I"]) * dt) / (2 * F)  # 4 cells in series carry the same current
    assert bench.vessel["stack_H2"][-1] == pytest.approx(n_from_current, rel=0.02)


def test_system_sankey_balances(bench):
    s = bench.sankey
    out_of = {i: 0.0 for i in range(len(s["nodes"]))}
    into = {i: 0.0 for i in range(len(s["nodes"]))}
    for l in s["links"]:
        out_of[l["source"]] += l["value"]
        into[l["target"]] += l["value"]
    names = {n: i for i, n in enumerate(s["nodes"])}
    for inner in ("H2 to stack", "Stack electricity"):
        assert out_of[names[inner]] == pytest.approx(into[names[inner]], rel=1e-6)
    assert out_of[names["H2 chemical energy (LHV)"]] > 0 and into[names["Electrical output"]] > 0


def test_starvation_warning_and_flags(bench):
    assert bench.summary["starve_s"] > 0 and any("starved" in w for w in bench.warnings)
    assert np.all((bench["phi"] >= 0) & (bench["phi"] <= 1.0 + 1e-9))


def test_power_load_with_battery_buffer_reduces_unmet_load():
    sc = Scenario(al_mass_g=5.0, mode="regulated", duration_s=1800.0, c_naoh_M=2.0, form="foil", dim_um=20.0,
                  load_steps=[(0.0, 0.0)])
    prof = [(0.0, 0.0), (200.0, 1.5), (900.0, 4.0), (1300.0, 1.0)]
    base = simulate_system(sc, SystemSpec(load_kind="power", load_power_W=prof))
    buf = simulate_system(sc, SystemSpec(load_kind="power", load_power_W=prof, battery_Wh=2.0, battery_p_max_W=10.0))
    assert buf.summary["Wh_unmet"] < base.summary["Wh_unmet"] or base.summary["Wh_unmet"] == 0.0
    assert 0.1 - 1e-9 <= buf["soc"].min() and buf["soc"].max() <= 0.95 + 1e-9
    assert buf.summary["buffer_Wh_needed"] >= 0.0


def test_system_model_class_reusable_and_deterministic():
    sc = benchmark_scenario(duration_s=600.0)
    a = SystemModel(sc, SystemSpec(seed=3)).run()
    b = SystemModel(sc, SystemSpec(seed=3)).run()
    assert np.allclose(a["V"], b["V"]) and a.summary["Wh_load"] == b.summary["Wh_load"]
