import time

import numpy as np
import pytest

from hydra.core import l1_fast as lf
from hydra.core.l1 import SolverSettings, simulate
from hydra.core.params import ParamSet
from hydra.core.scenario import Scenario
from hydra.surrogate import dataset, model

SPACE = dataset.InputSpace(duration_s=7200.0)


@pytest.fixture(scope="module")
def trained():
    d = dataset.generate_dataset(700, SPACE, "fast", workers=2, seed=1)
    s = model.Surrogate(SPACE)
    meta = s.fit(d)
    return d, s, meta


def sc_of(**kw):
    base = dict(al_mass_g=1.0, mode="open", duration_s=7200.0, dim_um=20.0, form="powder", c_naoh_M=4.0, v_liq_mL=200.0,
                vent_diameter_mm=10.0, v_vessel_mL=500.0)
    base.update(kw)
    return Scenario(**base)


def test_dataset_generation_is_parallel_consistent_and_filtered():
    a = dataset.generate_dataset(24, SPACE, "fast", workers=2, seed=5)
    b = dataset.generate_dataset(24, SPACE, "fast", workers=1, seed=5)
    assert np.allclose(a["X"], b["X"]) and a["X"].shape[1] == SPACE.n_time
    assert np.all(a["peak_T_C"] <= SPACE.peak_T_max_C) and np.all(np.isfinite(a["t_half"]))
    assert np.all((a["X"] >= 0) & (a["X"] <= 1)) and np.all(np.diff(a["X"], axis=1) >= -1e-9)
    assert len(a["x"]) <= a["n_requested"]
    full = dataset.generate_dataset(4, dataset.InputSpace(duration_s=3600.0), "L1", workers=1, seed=2)
    assert full["fidelity"] == "L1" and full["X"].shape[1] == 61


def test_error_bounds_hold_on_fresh_data_and_test_split(trained):
    d, s, meta = trained
    assert meta["coverage_test"]["X"] >= 0.8 and meta["coverage_test"]["dT"] >= 0.8 and meta["test_rmse_X"] < 0.05
    fresh = dataset.generate_dataset(160, SPACE, "fast", workers=2, seed=99)
    pred = s._curves(fresh["feat"])
    ex = np.max(np.abs(pred["X"] - fresh["X"]), axis=1)
    et = np.max(np.abs(pred["dT"] - fresh["dT"]), axis=1)
    assert np.mean(ex <= s.bounds["X"]) >= 0.85 and np.mean(et <= s.bounds["dT"]) >= 0.85  # stated 95 % conformal bounds
    assert s.bounds["X"] < 0.2


def test_physics_constraints_hold_everywhere_in_domain(trained):
    _, s, _ = trained
    rng = np.random.default_rng(0)
    lo, hi = SPACE.lo_hi()
    n_ok = 0
    for _ in range(60):
        x = lo + (hi - lo) * rng.random(len(lo))
        sc = dataset.scenario_of(x, SPACE)
        if not s.domain_check(sc)[0]:
            continue
        o = s.predict(sc, physics_fallback=False)
        nmax = 1.5 * sc.al_mass_g * 1e-3 / 26.9815385e-3
        assert o.h2_mol[0] == 0.0 and np.all(np.diff(o.h2_mol) >= -1e-15)
        assert o.h2_mol.max() <= min(1.0, sc.c_naoh_M * sc.v_liq_mL / 1e3 / (sc.al_mass_g * 1e-3 / 26.9815385e-3)) * nmax * (1 + 1e-9)  # mass conservation / NaOH limit
        assert o.T_K[0] == pytest.approx(sc.T0_C + 273.15)
        n_ok += 1
    assert n_ok >= 10


def test_stoichiometry_1g_al_in_surrogate(trained):
    _, s, _ = trained
    o = s.predict(sc_of())
    assert o.source == "surrogate" and o.in_domain
    assert o.h2_mol[-1] == pytest.approx(0.0556, abs=1e-4)
    assert o.h2_mol[-1] * 22.414 == pytest.approx(1.246, rel=2e-3)


def test_surrogate_speed_under_50ms(trained):
    _, s, _ = trained
    sc = sc_of(c_naoh_M=2.5, al_mass_g=1.5)
    s.predict(sc)
    ts = []
    for _ in range(15):
        t0 = time.perf_counter()
        s.predict(sc)
        ts.append(time.perf_counter() - t0)
    assert np.median(ts) < 0.05


def test_domain_of_validity_falls_back_to_physics(trained):
    _, s, _ = trained
    bad = sc_of(al_mass_g=20.0, c_naoh_M=0.5, v_liq_mL=100.0)  # outside box and NaOH-limited
    o = s.predict(bad)
    assert o.source == "physics" and not o.in_domain and o.reasons
    ref = lf.simulate_fast(lf.build_constants(bad, dt=4.0), bad, dt=4.0)
    assert o.h2_mol[-1] == pytest.approx(np.interp(7200.0, ref["t"], ref["gen"]), rel=1e-9)
    other_cfg = sc_of(mode="relief")
    assert s.predict(other_cfg).source == "physics"
    assert s.predict(bad, physics_fallback=False).source == "surrogate"  # caller can force it (and is warned by in_domain)


def test_surrogate_matches_full_model_within_stated_bound(trained):
    _, s, _ = trained
    for kw in (dict(), dict(c_naoh_M=2.0, al_mass_g=0.8, dim_um=40.0), dict(form="foil", dim_um=30.0, T0_C=30.0, T_amb_C=25.0, c_naoh_M=3.0)):
        sc = sc_of(**kw)
        o = s.predict(sc)
        if o.source != "surrogate":
            continue
        r = simulate(sc, settings=SolverSettings(dt_out=float(SPACE.t_grid[1])))
        nmax = 1.5 * sc.al_mass_g * 1e-3 / 26.9815385e-3
        err = np.max(np.abs(o.h2_mol / nmax - np.interp(o.t, r.t, r["h2_gen_mol"]) / nmax))
        assert err < s.bounds["X"] + 0.02  # + fast-vs-full model gap


def test_save_load_roundtrip(trained, tmp_path):
    _, s, _ = trained
    s.save(tmp_path / "s.pkl")
    s2 = model.Surrogate.load(tmp_path / "s.pkl")
    sc = sc_of(c_naoh_M=3.0)
    assert np.allclose(s.predict(sc).h2_mol, s2.predict(sc).h2_mol) and s2.meta["param_hash"] == s.meta["param_hash"]
    assert s.meta["param_hash"] != model.sha256(ParamSet({"k25": 1e-4}).v)[:12]


def test_snap_and_constraints_helpers():
    x = np.array([0.0, 0.3, 0.9, 0.989, 0.99, 0.99])
    out = model.snap_to_stoichiometry(x, 1.0)
    assert out[-1] == pytest.approx(1.0) and np.all(np.diff(out) >= 0) and out[0] == 0.0
    nl = np.array([0.0, 0.2, 0.4, 0.55, 0.7])
    assert np.allclose(model.snap_to_stoichiometry(nl, 1.0), nl)  # not saturated: untouched
    assert model.x_max_of(np.array([np.log(8.0), np.log(0.5), np.log(100.0)]))[0] < 0.6
    assert model.enforce_constraints(np.array([0.1, 0.5, 0.3, 1.4]))[-1] == 1.0


@pytest.mark.skipif(not lf.jax_available(), reason="JAX not installed")
def test_neural_ode_variant_is_bounded_monotone_and_learns():
    d = dataset.generate_dataset(60, dataset.InputSpace(duration_s=3600.0), "fast", workers=1, seed=3)
    n = model.NeuralODE(dataset.InputSpace(duration_s=3600.0), hidden=16)
    hist = n.fit(d, epochs=60, lr=1e-2)
    assert hist[-1] < hist[0]
    sc = sc_of(duration_s=3600.0)
    out = n.predict(sc)
    nmax = 1.5 * sc.al_mass_g * 1e-3 / 26.9815385e-3
    assert out["h2_mol"][0] == 0.0 and np.all(np.diff(out["h2_mol"]) >= -1e-12) and out["h2_mol"].max() <= nmax * (1 + 1e-9)
