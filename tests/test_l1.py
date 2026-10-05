import numpy as np
import pytest

from hydra.constants import MW_AL, P_ATM
from hydra.core import safety
from hydra.core.l1 import SolverSettings, simulate
from hydra.core.params import ParamSet
from hydra.core.scenario import Scenario, benchmark_scenario
from hydra.thermo import species


def excess(**kw):
    base = dict(al_mass_g=1.0, mode="open", duration_s=7200.0, dim_um=20.0, c_naoh_M=4.0)
    base.update(kw)
    return Scenario(**base)


@pytest.fixture(scope="module")
def stoich():
    return simulate(excess())


def test_stoichiometry_1g_al(stoich):
    n = 1.0e-3 / MW_AL * 1.5
    assert n == pytest.approx(0.0556, abs=1e-4)
    assert stoich.summary["h2_total_mol"] == pytest.approx(n, rel=1e-3)
    assert stoich.summary["h2_total_stp_L"] == pytest.approx(1.246, rel=2e-3)


@pytest.mark.parametrize("form,dim", [("powder", 100.0), ("wire", 200.0), ("can", 100.0), ("foil", 50.0)])
def test_stoichiometry_all_forms(form, dim):
    r = simulate(excess(form=form, dim_um=dim, duration_s=40000.0))
    assert r.summary["h2_total_stp_L"] == pytest.approx(1.246, rel=3e-3)


def test_conservation_all_atoms_charge_energy(stoich):
    for k in ("Al", "Na", "O", "H"):
        assert abs(stoich.ledger[k]) < 1e-3, k  # spec: < 0.1 %
    assert abs(stoich.ledger["charge"]) < 1e-9
    assert abs(stoich.ledger["energy"]) < 1e-3


def test_conservation_on_benchmark_with_valve_and_stack():
    r = simulate(benchmark_scenario())
    for k in ("Al", "Na", "O", "H", "energy"):
        assert abs(r.ledger[k]) < 1e-3, (k, r.ledger[k])
    assert r.summary["stack_h2_mol"] > 0
    assert r.diagnostics["n_valve_events"] > 0 and "t_valve_open" in r.events


def test_cumulative_h2_monotone_and_bounded(stoich):
    h = stoich["h2_gen_mol"]
    assert np.all(np.diff(h) >= -1e-12)
    assert h.max() <= stoich["h2_max_mol"][0] * (1 + 1e-6)


def test_limiting_reagent_naoh():
    # 0.1 M x 20 mL = 2 mmol OH-: at most 2 mmol Al reacts -> 3 mmol H2
    sc = excess(al_mass_g=1.0, c_naoh_M=0.1, v_liq_mL=20.0, duration_s=40000.0, v_vessel_mL=200.0)
    r = simulate(sc)
    assert r.summary["h2_total_mol"] <= 3.0e-3 * 1.001
    assert r.summary["h2_total_mol"] > 2.0e-3


def test_adiabatic_temperature_rise_matches_species_enthalpy_balance():
    sc = excess(al_mass_g=0.3, c_naoh_M=2.0, v_liq_mL=200.0, adiabatic=True, duration_s=3000.0,
                dim_um=10.0)
    r = simulate(sc, settings=SolverSettings(dt_out=5.0))
    # independent analytic: reaction heat / total heat capacity of final contents + wall
    n_al = 0.3e-3 / MW_AL
    T0 = 298.15
    dh = -species.reaction_enthalpy(T0)
    m = r.model
    n_sp = np.array([0.0, r.y[m.idx["nW"], -1], r.y[m.idx["nV"], -1], r.y[m.idx["nNa"], -1] - r.y[m.idx["nAlO"], -1],
                     r.y[m.idx["nNa"], -1], r.y[m.idx["nAlO"], -1], 0.0, r.y[m.idx["nH2"], -1],
                     r.y[m.idx["nH2d"], -1], r.y[m.idx["nAir"], -1]])
    from hydra.core.l1 import _CP
    c_final = float(n_sp @ _CP) + m.C_wall
    dT_analytic = dh * n_al / c_final
    dT_sim = r["T"][-1] - T0
    assert dT_sim == pytest.approx(dT_analytic, rel=0.03)
    assert r["Tw"][-1] == pytest.approx(r["T"][-1], abs=0.5)  # wall equilibrates adiabatically


def test_relief_valve_limits_pressure_and_hysteresis_cycles():
    r = simulate(benchmark_scenario(mode="relief"))
    pg = (r["P"] - P_ATM) / 1e5
    assert pg.max() <= 1.5 * 1.02
    assert r.diagnostics["n_valve_events"] >= 2


def test_sealed_vessel_pressure_rises_without_limit_flag():
    sc = excess(mode="sealed", al_mass_g=0.5, duration_s=3600.0, v_vessel_mL=500.0)
    r = simulate(sc)
    assert (r["P"][-1] - P_ATM) > 1.0e5
    assert safety.screen(sc).unsafe  # pre-run screening flags sealed-without-relief


def test_eos_deviation_small_at_low_pressure_and_visible_at_high():
    sc = excess(mode="sealed", al_mass_g=1.0, duration_s=3600.0, v_vessel_mL=300.0, v_liq_mL=200.0)
    pi = simulate(sc)["P"][-1]
    pa = simulate(excess(mode="sealed", al_mass_g=1.0, duration_s=3600.0, v_vessel_mL=300.0, v_liq_mL=200.0,
                         eos="abel_noble"))["P"][-1]
    assert pa > pi and (pa - pi) / pi < 0.1


def test_temperature_dependence_of_rate():
    t90 = []
    for T0 in (20.0, 40.0):
        t90.append(simulate(excess(T0_C=T0, T_amb_C=T0, c_naoh_M=2.0, dim_um=20.0)).summary["t90_s"])
    assert t90[1] < t90[0]


def test_rate_scales_with_naoh_concentration():
    t90 = [simulate(excess(c_naoh_M=c, al_mass_g=0.2, v_liq_mL=200.0)).summary["t90_s"] for c in (1.0, 4.0)]
    assert t90[1] < t90[0]


def test_boiling_in_open_vessel_clamps_temperature():
    r = simulate(excess(al_mass_g=5.0, c_naoh_M=2.0, v_liq_mL=100.0, adiabatic=True, dim_um=5.0,
                        duration_s=1800.0, v_vessel_mL=500.0, vent_diameter_mm=30.0))
    assert r["boil_mol"][-1] > 0
    assert r["T"].max() < 373.15 + 6.0


def test_dosing_changes_state_consistently():
    from hydra.core.l1 import L1Model
    m = L1Model(excess(mode="relief", al_mass_g=1.0))
    y = m.initial_state()
    m.u = (0.05, 0.0)  # mol/s water
    t, ys = m.integrate(y, 0.0, 20.0)
    assert ys[m.idx["nW"], -1] == pytest.approx(y[m.idx["nW"]] + 1.0, rel=0.01)
    assert ys[m.cidx["dose_w"], -1] == pytest.approx(1.0, rel=1e-6)


def test_safety_screen_and_analysis_flags():
    rep = safety.screen(benchmark_scenario())
    assert rep.unsafe and any(f.code == "ADIABATIC_T" for f in rep.flags)
    r = simulate(benchmark_scenario())
    post = safety.analyze(r)
    assert any(f.code in ("HDPE_T", "HOOP_SF", "SEMENOV", "FAST_HEATING") for f in post.flags)
    assert "NOT a safety certification" in post.disclaimer
    assert safety.time_to_limit(np.array([0, 10.0]), np.array([0, 1.0]), 11.0) == pytest.approx(100.0)
    assert safety.time_to_limit(np.array([0, 10.0]), np.array([1.0, 0.5]), 11.0) == float("inf")


def test_range_violations_reported_not_silent():
    r = simulate(benchmark_scenario())
    assert any("pitzer" in v for v in r.diagnostics["range_violations"])


def test_param_override_changes_rate():
    base = simulate(excess(al_mass_g=0.3, c_naoh_M=2.0))
    fast = simulate(excess(al_mass_g=0.3, c_naoh_M=2.0), ParamSet({"k25": 4e-4}))
    assert fast.summary["t90_s"] < base.summary["t90_s"]


def test_open_vessel_never_pulls_vacuum_and_ledger_closes() -> None:
    """An open vent must draw ambient air back in as the vessel cools (regression: P fell to 0.87 bar abs)."""
    sc = Scenario(al_mass_g=0.4, form="foil", dim_um=5, c_naoh_M=4, v_liq_mL=30, T0_C=23, T_amb_C=23, v_vessel_mL=250,
                  mode="open", cells=0, duration_s=1800)
    r = simulate(sc)
    p_bar = r.series["P"] / 1.0e5
    assert p_bar.min() > 1.0132 - 2e-3 and p_bar.max() < 1.0132 + 2e-3
    assert abs(r.ledger["energy"]) < 1e-3
    assert r.summary["conversion_pct"] > 99.0
