
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from hydra.thermo import electrolyte, gas, pitzer, propdb, solubility, species, water
from hydra.thermo.ranges import RangeError, RangeWarning, collect


def test_every_db_entry_has_source_unit_status():
    for e in propdb.table():
        assert e["source"] and e["unit"] and e["status"] in {"prior", "unverified", "exact"}, e["name"]


def test_reaction_enthalpy_and_hess_gibbs_closure():
    assert species.reaction_enthalpy(298.15) == pytest.approx(-415.0e3, rel=0.005)
    dg_f, dg_hs, diff = species.gibbs_consistency_298()
    assert abs(diff) / abs(dg_f) < 0.002  # Hess / Gibbs-Helmholtz closure < 0.2 %


def test_kirchhoff_slope_equals_delta_cp():
    dcp = species.reaction_delta_cp()
    num = (species.reaction_enthalpy(350.0) - species.reaction_enthalpy(300.0)) / 50.0
    assert num == pytest.approx(dcp, rel=1e-9)


def test_water_vapour_pressure_and_boiling():
    assert water.psat(298.15) == pytest.approx(3167.0, rel=0.01)
    assert water.tboil(101325.0) == pytest.approx(373.15, abs=0.2)
    assert water.hvap(373.15) == pytest.approx(40.66e3, rel=0.02)  # Watson correlation error


@given(st.floats(280.0, 370.0), st.floats(0.1, 9.0))
@settings(deadline=None, max_examples=50)
def test_psat_monotone_in_T(T, dT):
    assert water.psat(T + dT) > water.psat(T)


@given(st.floats(0.05, 5.5), st.floats(0.01, 0.5))
@settings(deadline=None, max_examples=50)
def test_density_and_viscosity_monotone_in_concentration(c, dc):
    assert electrolyte.density(c + dc, 298.15) > electrolyte.density(c, 298.15)
    assert electrolyte.viscosity(c + dc, 298.15) > electrolyte.viscosity(c, 298.15)


def test_pitzer_naoh_reference_values():
    # literature (approximate): NaOH 1 mol/kg gamma+- ~ 0.68; 2 mol/kg a_w ~ 0.93
    assert pitzer.mean_gamma_naoh(1.0) == pytest.approx(0.68, rel=0.03)
    assert pitzer.water_activity(2.0) == pytest.approx(0.93, abs=0.01)
    assert pitzer.osmotic_coefficient(1e-6) == pytest.approx(1.0, abs=1e-3)


def test_debye_hueckel_fallback_close_in_dilute_limit():
    assert pitzer.water_activity(0.01, model="debye_huckel") == pytest.approx(pitzer.water_activity(0.01), abs=1e-4)


def test_solubility_increases_with_temperature_and_naoh():
    with collect("ignore"):
        s = [solubility.equilibrium_aluminate(2.0, T) for T in (298.15, 323.15, 343.15)]
        assert s[0] < s[1] < s[2]
        assert solubility.equilibrium_aluminate(4.0, 298.15) > solubility.equilibrium_aluminate(1.0, 298.15)
        assert solubility.supersaturation(1.0, 0.0, 298.15) == 0.0
        m_sat = solubility.equilibrium_aluminate(2.0, 298.15)
        assert solubility.supersaturation(2.0 - m_sat, m_sat, 298.15) == pytest.approx(1.0, rel=1e-4)


def test_bayerite_more_soluble_than_gibbsite():
    assert solubility.equilibrium_aluminate(2.0, 298.15, "bayerite") > solubility.equilibrium_aluminate(2.0, 298.15)


def test_cnt_nucleation_needs_supersaturation_and_rises_steeply():
    assert solubility.nucleation_rate(0.9, 298.15) == 0.0
    assert solubility.nucleation_rate(8.0, 298.15) > 1e3 * solubility.nucleation_rate(4.0, 298.15)


def test_range_guard_warn_collect_and_error_policy():
    with collect() as box:
        pitzer.water_activity(1.0, T=350.0)
    assert any("pitzer" in k for k in box)
    with pytest.raises(RangeError):
        with collect("error"):
            pitzer.water_activity(1.0, T=350.0)
    with pytest.warns(RangeWarning):
        pitzer.water_activity(1.0, T=350.0)


def test_eos_ordering_and_ideal_limit():
    assert gas.compressibility_h2(1e-3, 1e-3, 300.0, "ideal") == pytest.approx(1.0)
    assert gas.compressibility_h2(1.0, 1e-3, 300.0, "abel_noble") > 1.0
    assert gas.compressibility_h2(1.0, 1e-3, 300.0, "peng_robinson") > 1.0
    assert gas.pressure_h2(1e-6, 1e-3, 300.0, "peng_robinson") == pytest.approx(
        gas.pressure_h2(1e-6, 1e-3, 300.0, "ideal"), rel=1e-4)


def test_henry_decreases_with_naoh_and_matches_reference():
    assert gas.henry_h2(298.15, 2.0) < gas.henry_h2(298.15, 0.0)
    assert gas.henry_h2(298.15) == pytest.approx(7.8e-6, rel=1e-6)


def test_density_closed_form_consistency():
    rho, c = electrolyte.density_from_mass(0.4, 0.216, 298.15)
    assert rho == pytest.approx(electrolyte.density(c, 298.15), rel=1e-9)


def test_validation_helper(tmp_path):
    f = tmp_path / "t.csv"
    f.write_text("property,T_K,c_mol_L,value\ndensity,298.15,2.0,1080\ndensity,298.15,1.0,1040\n")
    out = __import__("hydra.thermo.validate", fromlist=["x"])
    res = out.validate(out.load_table(f))
    assert res["density"]["n"] == 2 and res["density"]["max_abs_rel"] < 0.02
