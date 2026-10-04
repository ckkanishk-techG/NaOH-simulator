import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hydra.constants import T_REF, F, R
from hydra.core.l1 import simulate
from hydra.core.params import ParamSet
from hydra.core.rates import ArrheniusModel
from hydra.core.scenario import Scenario
from hydra.electrochem.model import ElectrochemModel, effective_arrhenius, polarization_curve, tafel_limit

IDEAL = ParamSet({"ec_use_activities": 0.0})


def test_reduces_to_arrhenius_across_T_and_c_grid():
    m, a = ElectrochemModel(IDEAL, 1.0), ArrheniusModel(IDEAL, 1.0)
    for T in (285.0, 298.15, 320.0, 350.0):
        for c in (0.3, 1.0, 2.0, 4.0):
            m.ctx = (c, 0.0)
            j, _ = m.flux(T, c, 1.0, 0.0, 0.0)
            ja, _ = a.flux(T, c, 1.0, 0.0, 0.0)
            assert j == pytest.approx(ja, rel=2e-3), (T, c)


def test_effective_arrhenius_recovers_prior_parameters():
    fit = effective_arrhenius(ElectrochemModel(IDEAL, 1.0))
    assert fit["k25"] == pytest.approx(IDEAL["k25"], rel=5e-3)
    assert fit["Ea"] == pytest.approx(IDEAL["Ea"], rel=5e-3)
    assert fit["n"] == pytest.approx(IDEAL["n_oh"], abs=5e-3)


def test_L1_run_with_fast_film_matches_empirical_model():
    base = dict(al_mass_g=1.0, mode="open", duration_s=3600.0, dim_um=20.0, c_naoh_M=2.0)
    arr = simulate(Scenario(**base), IDEAL.with_(tau_ind=1.0e-2, ec_aAl_floor=50.0))
    ec = simulate(Scenario(rate_model="electrochemical", **base), IDEAL.with_(ec_k_diss=1.0e3, ec_k_pass=0.0, ec_aAl_floor=50.0))
    for k in (0.25, 0.5, 0.75):
        i = int(k * len(arr.t))
        assert ec["h2_gen_mol"][i] == pytest.approx(arr["h2_gen_mol"][i], rel=0.02)
    assert ec.summary["peak_T_C"] == pytest.approx(arr.summary["peak_T_C"], abs=0.5)


def test_numeric_mixed_potential_matches_tafel_closed_form():
    m = ElectrochemModel(IDEAL, 1.0)
    for T, a in ((298.15, 1.0), (330.0, 3.0)):
        m.ctx = (a, 0.0)
        _, i = m.couple(T, a, 0.0)
        assert i == pytest.approx(tafel_limit(m, T, a, 0.0), rel=2e-2)


def test_faraday_relation_h2_from_corrosion_current():
    m = ElectrochemModel(IDEAL, 1.0)
    m.ctx = (2.0, 0.0)
    _, i = m.couple(300.0, 2.0, 0.0)
    j, _ = m.flux(300.0, 2.0, 1.0, 0.0, 0.0)
    assert 1.5 * j == pytest.approx(i / (2.0 * F), rel=1e-9)  # n_H2 = i/2F = 1.5 n_Al
    assert j == pytest.approx(i / (3.0 * F), rel=1e-9)


@given(st.floats(0.2, 5.0), st.floats(280.0, 360.0))
@settings(deadline=None, max_examples=40)
def test_monotone_in_concentration_and_temperature(c, T):
    m = ElectrochemModel(IDEAL, 1.0)
    m.ctx = (c, 0.0)
    j0, _ = m.flux(T, c, 1.0, 0.0, 0.0)
    m.ctx = (c * 1.2, 0.0)
    j1, _ = m.flux(T, c * 1.2, 1.0, 0.0, 0.0)
    m.ctx = (c, 0.0)
    j2, _ = m.flux(T + 5.0, c, 1.0, 0.0, 0.0)
    assert j1 > j0 and j2 > j0


def test_corrosion_potential_between_equilibrium_potentials():
    m = ElectrochemModel(IDEAL, 1.0)
    pc = polarization_curve(m, 298.15, 2.0)
    assert pc["E_a"] < pc["E_corr"] < pc["E_c"]
    assert pc["i_corr"] > 0
    # partial currents cross at E_corr
    k = int(np.argmin(np.abs(pc["i_anodic"] - pc["i_cathodic"])))
    assert abs(pc["E"][k] - pc["E_corr"]) < 0.05


def test_aluminate_accumulation_lowers_rate():
    m = ElectrochemModel(IDEAL, 1.0)
    m.ctx = (2.0, 0.0)
    _, i0 = m.couple(298.15, 2.0, 0.0)
    _, i1 = m.couple(298.15, 2.0, 1.5)
    assert i1 < i0


def test_oxide_film_induction_period_and_monotone_activation():
    sc = Scenario(al_mass_g=1.0, mode="open", duration_s=1500.0, dim_um=20.0, c_naoh_M=2.0,
                  rate_model="electrochemical")
    r = simulate(sc)
    assert r["film"][0] == 0.0 and np.all(np.diff(r["film"]) >= -1e-9)
    assert r["film"][-1] > 0.9
    assert r["flow"][5] < 0.2 * r["flow"].max()  # induction: slow start


def test_stalling_at_low_hydroxide_matches_film_steady_state():
    sc = Scenario(al_mass_g=1.0, mode="open", duration_s=7200.0, dim_um=20.0, c_naoh_M=0.01,
                  v_liq_mL=200.0, rate_model="electrochemical")
    p = ParamSet()
    r = simulate(sc, p)
    kd = p["ec_k_diss"] * 0.01 * 1.0  # a_OH ~ c at 0.01 M (gamma ~ 1)
    psi_inf = kd / (kd + p["ec_k_pass"])
    assert r["film"][-1] == pytest.approx(psi_inf, rel=0.15)
    assert r.summary["conversion_pct"] < 10.0  # stalled


def test_galvanic_contact_accelerates():
    base = dict(al_mass_g=0.5, mode="open", duration_s=3000.0, dim_um=20.0, c_naoh_M=2.0, rate_model="electrochemical")
    slow = simulate(Scenario(**base))
    fast = simulate(Scenario(**base), ParamSet({"ec_galv_ratio": 1.0}))
    assert fast.summary["t90_s"] < slow.summary["t90_s"]


def test_electrochemical_stoichiometry_and_conservation():
    r = simulate(Scenario(al_mass_g=1.0, mode="open", duration_s=7200.0, dim_um=20.0, c_naoh_M=4.0,
                          rate_model="electrochemical"))
    assert r.summary["h2_total_stp_L"] == pytest.approx(1.246, rel=2e-3)
    assert all(abs(r.ledger[k]) < 1e-3 for k in ("Al", "Na", "O", "H", "energy"))
    assert "E_corr" in r.series and r["E_corr"][len(r.t) // 2] < -1.0


def test_arrhenius_temperature_dependence_of_icorr_is_arrhenius_like():
    m = ElectrochemModel(IDEAL, 1.0)
    ts = np.linspace(290.0, 340.0, 9)
    ln_i = []
    for T in ts:
        m.ctx = (1.0, 0.0)
        ln_i.append(math.log(m.flux(float(T), 1.0, 1.0, 0.0, 0.0)[0]))
    slope, icpt = np.polyfit(1.0 / ts, ln_i, 1)
    pred = slope / ts * 0 + np.polyval([slope, icpt], 1.0 / ts)
    assert np.max(np.abs(pred - ln_i)) < 0.01  # linear in 1/T
    assert -slope * R == pytest.approx(IDEAL["Ea"], rel=0.03)
    _ = T_REF
