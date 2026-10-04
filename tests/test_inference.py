import json

import numpy as np
import pytest

from hydra.core import l1_fast as lf
from hydra.core.l1 import SolverSettings, simulate
from hydra.core.params import ParamSet
from hydra.core.scenario import Scenario
from hydra.inference import bayes, benchmarks, discrepancy, hierarchical, modelsel, sensitivity, validation
from hydra.inference.calibrate import Predictor, fisher_information, fit_lsq, profile_likelihood
from hydra.inference.corrections import (Reading, correction_table, mass_loss_estimate_vapor, moles_from_reading,
                                         moles_with_uncertainty)
from hydra.inference.data import Channel, Experiment, ParamSpace, synthetic_experiment
from hydra.inference.reconcile import bias_factor, reconcile, reconcile_curves
from hydra.inference.sensors import Sensor

TRUTH = ParamSet({"k25": 3.0e-4, "Ea": 52e3, "n_oh": 0.9, "tau_ind": 40.0})
NAMES = ["k25", "Ea", "n_oh", "tau_ind"]


def make_exps(rng, conds=((1.0, 25, 20), (2.0, 25, 20), (3.0, 35, 20), (2.0, 20, 40)), dur=1500.0, dt=2.0, truth=TRUTH,
              model="empirical", **kw):
    out = []
    for i, (c, t0, d) in enumerate(conds):
        sc = Scenario(al_mass_g=1.0, mode="open", duration_s=dur, dim_um=d, c_naoh_M=c, T0_C=t0, T_amb_C=t0)
        out.append(synthetic_experiment(f"e{i}", sc, truth, rng, model=model, dt=dt, **kw))
    return out


# ---------------------------------------------------------------- measurement science
def test_sensor_lag_gain_offset_drift_and_deconvolution():
    t = np.arange(0.0, 600.0, 2.0)
    true = 25.0 + 30.0 * (1 - np.exp(-t / 200.0))
    s = Sensor(tau_s=15.0, gain=1.01, offset=0.2, drift_per_h=0.5)
    y = s.apply(t, true)
    assert np.max(np.abs(y - true)) > 1.0  # lag matters
    back = s.invert(t, y, smooth=0.01)
    assert np.max(np.abs(back - true)[5:-5]) < 1e-3
    nonuni = np.cumsum(np.r_[0, np.random.default_rng(0).uniform(1.9, 2.1, len(t) - 1)])
    assert np.max(np.abs(Sensor(tau_s=15.0).lag(nonuni, 25 + 0.02 * nonuni) - (25 + 0.02 * nonuni - 0.02 * 15.0))[100:]) < 5e-3


def test_sensor_sigma_combines_noise_resolution_calibration():
    s = Sensor(noise_sd=0.1, resolution=0.3, unc_gain_rel=0.01, unc_offset=0.05)
    sig = s.sigma(np.array([10.0]))[0]
    assert sig == pytest.approx(np.sqrt(0.1**2 + 0.3**2 / 12 + 0.1**2 + 0.05**2))
    r = s.simulate(np.arange(5.0), np.arange(5.0), np.random.default_rng(0))
    assert np.allclose(r / 0.3, np.round(r / 0.3))  # quantised


def test_gas_volume_corrections():
    r = Reading("water_displacement", 1.0, T_K=295.15, head_m=0.02, contact_water_L=0.2)
    tab = correction_table(r)
    assert tab["relative_correction"] < -0.01  # vapour pressure removes ~2.6 %
    assert tab["vapour_pressure_Pa"] == pytest.approx(2634.8, rel=0.01)
    assert tab["hydrostatic_Pa"] < 0  # inside level above outside -> lower pressure
    dry = moles_from_reading(Reading("gas_syringe", 1.0, T_K=295.15, rh=0.0))
    assert dry == pytest.approx(101325.0 * 1e-3 / (8.314462618 * 295.15), rel=1e-9)
    assert moles_from_reading(Reading("flowmeter", 1.0)) == pytest.approx(1.0 / 22.414, rel=1e-3)
    assert moles_from_reading(Reading("mass_loss", 0.2016)) == pytest.approx(0.1, rel=1e-6)
    m, sd = moles_with_uncertainty(Reading("gas_syringe", 1.0, unc={"value": 0.01, "T_K": 0.5}), n_mc=500)
    assert 0 < sd / m < 0.03
    assert mass_loss_estimate_vapor(0.1, 330.0) > 0


def test_reconciliation_flags_gross_error_and_bias():
    rec = reconcile(np.array([0.050, 0.051, 0.049, 0.065]), np.array([0.002] * 4))
    assert rec.flagged == [3] and rec.estimate == pytest.approx(0.05, abs=1e-3)
    ok = reconcile(np.array([0.050, 0.051, 0.049]), np.array([0.002] * 3))
    assert ok.flagged == [] and ok.p_value > 0.05
    t = np.linspace(0, 100, 21)
    curves = [(t, 0.001 * t, np.full_like(t, 0.002)), (t, 0.001 * t + 0.0005, np.full_like(t, 0.002)),
              (t, 0.001 * t + 0.05, np.full_like(t, 0.002))]
    out = reconcile_curves(t[5:], curves)
    assert np.all(out["n_flagged"] == 1) and np.allclose(out["n"], 0.001 * t[5:], atol=1e-3)
    beta, se = bias_factor(0.001 * t[1:], 1.05 * 0.001 * t[1:], np.full(20, 1e-3), np.full(20, 1e-3))
    assert beta == pytest.approx(1.05) and se > 0


# ---------------------------------------------------------------- fast model cross-check
def test_fast_model_agrees_with_full_adaptive_model():
    sc = Scenario(al_mass_g=0.5, mode="open", duration_s=3600.0, dim_um=20.0, c_naoh_M=2.0, vent_diameter_mm=30.0)
    full = simulate(sc, settings=SolverSettings(dt_out=2.0))
    fast = lf.simulate_fast(lf.build_constants(sc, dt=1.0), sc, dt=1.0)
    n = np.interp(full.t, fast["t"], fast["gen"])
    assert np.max(np.abs(n - full["h2_gen_mol"])) / full["h2_gen_mol"][-1] < 0.01
    tf = np.interp(full.t, fast["t"], fast["T"])
    assert np.max(np.abs(tf - full["T"])) < 0.5


def test_fast_model_stoichiometry_and_applicability():
    sc = Scenario(al_mass_g=1.0, mode="open", duration_s=20000.0, dim_um=20.0, c_naoh_M=4.0)
    s = lf.simulate_fast(lf.build_constants(sc), sc)
    assert s["gen"][-1] == pytest.approx(1.5 * 1e-3 / 26.9815385e-3, rel=2e-3)
    assert lf.fast_applicable(sc) == []
    assert lf.fast_applicable(sc.model_copy(update={"mode": "relief"}))


@pytest.mark.skipif(not lf.jax_available(), reason="JAX not installed")
def test_jax_and_numpy_paths_agree_and_gradients_are_exact():
    sc = Scenario(al_mass_g=0.5, mode="open", duration_s=900.0, dim_um=20.0, c_naoh_M=2.0)
    for model in ("empirical", "mass_transfer", "echem"):
        k = lf.build_constants(sc, model=model)
        a, b = lf.simulate_fast(k, sc, 3.0), lf.simulate_fast(k, sc, 3.0, backend="jax")
        for key in ("gen", "T", "Tw", "film"):
            assert np.max(np.abs(a[key] - b[key]) / np.maximum(np.abs(a[key]), 1e-12)) < 1e-6  # spec: 1e-6 relative
    from hydra.inference.jaxlik import gradient_of_outputs

    g = gradient_of_outputs(sc, ["k25", "Ea", "n_oh"], dt=3.0)
    base = ParamSet()
    for i, n in enumerate(["k25", "Ea", "n_oh"]):
        h = 1e-5 * base[n]
        up = lf.simulate_fast(lf.build_constants(sc, base.with_(**{n: base[n] + h}), dt=3.0), sc, 3.0)["gen"][-1]
        dn = lf.simulate_fast(lf.build_constants(sc, base.with_(**{n: base[n] - h}), dt=3.0), sc, 3.0)["gen"][-1]
        assert g["jac"][0, i] == pytest.approx((up - dn) / (2 * h), rel=1e-4, abs=1e-14)


# ---------------------------------------------------------------- calibration
@pytest.fixture(scope="module")
def calib():
    rng = np.random.default_rng(1)
    exps = make_exps(rng)
    pred = Predictor(exps)
    fit = fit_lsq(pred, NAMES)
    return exps, pred, fit


def test_lsq_recovers_synthetic_parameters_within_uncertainty(calib):
    _, _, fit = calib
    sd = fit.sd()
    for n in NAMES:
        assert abs(fit.theta[n] - TRUTH[n]) < 3.5 * sd[n], (n, fit.theta[n], TRUTH[n], sd[n])
    assert 0.7 < fit.red_chi2 < 1.4
    assert fit.aic < fit.bic + 100 and fit.dof == fit.n - 4


def test_fisher_information_and_identifiability(calib):
    exps, pred, fit = calib
    fi = fisher_information(fit)
    assert fi["cond"] > 1 and fi["eigvals"].min() > 0 and fi["unidentifiable"] == []
    # one isothermal experiment cannot identify Ea separately from k25: flagged
    one = Predictor([make_exps(np.random.default_rng(3), conds=((2.0, 25, 20),))[0]])
    f1 = fit_lsq(one, ["k25", "Ea", "n_oh"])
    assert fisher_information(f1)["cond"] > 100 * fi["cond"] or "Ea" in fisher_information(f1)["unidentifiable"]


def test_profile_likelihood_interval_contains_truth(calib):
    _, pred, fit = calib
    pr = profile_likelihood(pred, fit, "Ea", n_grid=11)
    lo, hi = pr["ci95"]
    assert lo < TRUTH["Ea"] * 1.002 and hi > TRUTH["Ea"] * 0.998 and not pr["flat"]
    assert pr["delta"].min() >= -1e-6


def test_param_space_roundtrip_and_shear():
    exps = make_exps(np.random.default_rng(0))
    sp = ParamSpace.for_experiments(NAMES, exps)
    th = {"k25": 2.5e-4, "Ea": 48000.0, "n_oh": 0.85, "tau_ind": 50.0}
    back = sp.from_u(sp.to_u(th))
    assert all(back[k] == pytest.approx(th[k], rel=1e-10) for k in th)
    assert np.isfinite(sp.log_prior(sp.to_u(th))) and sp.log_prior(sp.to_u({**th, "n_oh": 5.0})) == -np.inf


# ---------------------------------------------------------------- bayesian
def test_stretch_move_samples_a_gaussian():
    target = lambda x: -0.5 * np.sum(((x - np.array([1.0, -2.0])) / np.array([0.5, 2.0])) ** 2)  # noqa: E731
    p0 = np.random.default_rng(0).standard_normal((16, 2))
    ch, _, acc = bayes.stretch_move(target, p0, 1500, seed=1)
    s = ch[500:].reshape(-1, 2)
    assert s.mean(axis=0) == pytest.approx([1.0, -2.0], abs=0.15) and s.std(axis=0) == pytest.approx([0.5, 2.0], rel=0.15)
    assert 0.2 < acc < 0.9 and bayes.autocorr_time(ch[500:, :, 0].mean(axis=1)) > 0


def test_posterior_credible_intervals_contain_truth_and_ppc(calib):
    exps, pred, fit = calib
    post = bayes.Posterior(pred, NAMES)
    ch = bayes.sample(post, post.start(fit), nsteps=500, burn=250, seed=0, backend="stretch")
    ci = bayes.credible_intervals(ch, 0.99)
    for n in NAMES:
        assert ci[n]["lo"] <= TRUTH[n] <= ci[n]["hi"], (n, ci[n])
    cov = bayes.ppc_coverage(ch, 60)["coverage95"]
    assert 0.88 < cov <= 1.0
    w = bayes.waic(ch, 60)
    assert np.isfinite(w["waic"]) and 2 < w["p_waic"] < 8
    cd = bayes.corner_data(ch)
    assert cd["corr"].shape == (4, 4) and (("Ea", "k25") in cd["pairs"] or ("k25", "Ea") in cd["pairs"])
    bands = bayes.posterior_predictive(ch, 30)
    b = bands[0][0]
    assert np.all(b["lo"] <= b["med"]) and np.all(b["med"] <= b["hi"]) and np.all(b["pred_lo"] <= b["lo"] + 1e-9)


# ---------------------------------------------------------------- sensitivity + selection
def test_sobol_and_tornado():
    sc = Scenario(al_mass_g=1.0, mode="open", duration_s=1200.0, dim_um=20.0, c_naoh_M=2.0)
    res = sensitivity.sobol_indices(sc, ["k25", "Ea", "n_oh", "tau_ind"], n=64, qoi="h2_mol_at_tref")
    assert sum(res["S1"].values()) < 1.3 and all(v > -0.3 for v in res["ST"].values())
    assert max(res["ST"], key=res["ST"].get) in ("k25", "Ea", "n_oh")
    assert res["ST"]["k25"] > res["ST"]["tau_ind"] * 0.9
    tor = sensitivity.tornado(sc, ["k25", "Ea", "n_oh", "tau_ind"])
    assert tor[0]["swing"] >= tor[-1]["swing"] and tor[0]["name"] in ("k25", "Ea", "n_oh")


def test_model_selection_prefers_the_generating_model():
    rng = np.random.default_rng(5)
    mt_truth = TRUTH.with_(k_mt=2.0e-6)
    exps_mt = make_exps(rng, truth=mt_truth, model="mass_transfer", dur=3000.0, conds=((1.0, 25, 20), (2.0, 25, 20), (3.0, 35, 20), (4.0, 30, 20)))
    tab = modelsel.compare_models(exps_mt, ["empirical", "mass_transfer"])
    assert tab[0]["model"] == "mass_transfer" and tab[1]["dBIC"] > 10
    exps_emp = make_exps(np.random.default_rng(6))
    tab2 = modelsel.compare_models(exps_emp, ["empirical", "mass_transfer"])
    assert tab2[0]["model"] == "empirical"


# ---------------------------------------------------------------- discrepancy / hierarchical
def test_discrepancy_gp_detects_missing_physics_and_ignores_noise():
    rng = np.random.default_rng(7)
    # truth has oxide-film stalling at low OH- (echem), the fitted empirical model cannot reproduce it
    exps = make_exps(rng, truth=ParamSet({"ec_k_diss": 2e-3, "ec_k_pass": 3e-3, "k25": 3e-4, "Ea": 52e3, "n_oh": 0.9}),
                     model="echem", dur=2400.0, conds=((0.3, 25, 20), (1.0, 25, 20), (2.0, 25, 20), (4.0, 25, 20)), kinds=("n_h2",))
    pred = Predictor(exps)
    fit = fit_lsq(pred, NAMES)
    tab = discrepancy.residual_table(pred, pred.base.with_(**fit.theta))
    gp = discrepancy.DiscrepancyGP(tab)
    rep = discrepancy.discrepancy_report(tab, gp)
    assert rep["overall_rms_delta"] > 0.5 * rep["median_sigma"]
    assert set(rep["by_feature"]) == set(discrepancy.FEATURES) and "feature" in rep["worst_region"]
    # clean data: discrepancy is negligible next to the noise
    clean = make_exps(np.random.default_rng(8), kinds=("n_h2",))
    p2 = Predictor(clean)
    f2 = fit_lsq(p2, NAMES)
    t2 = discrepancy.residual_table(p2, p2.base.with_(**f2.theta))
    r2 = discrepancy.discrepancy_report(t2, discrepancy.DiscrepancyGP(t2))
    assert r2["overall_rms_delta"] < 1.0 * r2["median_sigma"]


def test_koh_calibration_runs_and_returns_gp():
    exps = make_exps(np.random.default_rng(9), conds=((1.0, 25, 20), (3.0, 30, 20)), dur=900.0, kinds=("n_h2",))
    pred = Predictor(exps)
    out = discrepancy.koh_calibrate(pred, ["k25", "n_oh"], iters=1)
    assert set(out["theta"]) == {"k25", "n_oh"} and np.isfinite(out["nlml_history"][0])


def test_hierarchical_recovers_batch_variation():
    rng = np.random.default_rng(11)
    exps = []
    mult = {"A": 0.75, "B": 1.0, "C": 1.35}
    for b, m in mult.items():
        for i, (c, t0) in enumerate(((1.0, 25), (3.0, 30))):
            sc = Scenario(al_mass_g=1.0, mode="open", duration_s=900.0, dim_um=20.0, c_naoh_M=c, T0_C=t0, T_amb_C=t0)
            e = synthetic_experiment(f"{b}{i}", sc, TRUTH.with_(k25=TRUTH["k25"] * m), rng, dt=4.0, batch=b, kinds=("n_h2",))
            exps.append(e)
    post = hierarchical.HierPosterior(exps, ["k25", "n_oh"], ["k25"], dt=4.0)
    th0 = {"k25": TRUTH["k25"], "n_oh": 0.9}
    res = hierarchical.sample_hier(post, post.start(th0), nsteps=300, burn=150, seed=1)
    ua, ub, uc = (np.mean(res.batch_effect["k25"][b]) for b in "ABC")
    # only differences between batches are identifiable (the mean is traded against the shared k25)
    assert ub - ua == pytest.approx(np.log(1.0 / 0.75), abs=0.15) and uc - ub == pytest.approx(np.log(1.35), abs=0.15)
    assert 0.05 < np.median(res.tau["k25"]) < 1.0  # batch variation detected, tau not collapsed to zero


# ---------------------------------------------------------------- validation protocol
def test_split_leakage_and_blind_guard():
    rng = np.random.default_rng(0)
    exps = make_exps(rng)
    exps[0].split = "blind"
    with pytest.raises(ValueError):
        validation.assert_no_leakage(exps, [])
    cal = [e for e in exps if e.split == "calib"]
    validation.assert_no_leakage(cal, [exps[0]])
    with pytest.raises(ValueError):
        validation.assert_no_leakage(cal, [cal[0]])
    assert {k: len(v) for k, v in validation.split_experiments(exps).items()} == {"calib": 3, "val": 0, "blind": 1}


def test_metrics_reliability_and_overconfidence_flag():
    t = np.linspace(0, 100, 50)
    y = 0.001 * t
    rng = np.random.default_rng(0)
    obs = y + 0.01 * rng.standard_normal(50)
    m = validation.point_metrics(t, obs, y)
    assert m["rmse"] == pytest.approx(0.01, rel=0.4) and m["mape"] > 0
    good = validation.reliability(obs, y, np.full(50, 0.01))
    bad = validation.reliability(obs, y, np.full(50, 0.002))
    assert validation.calibration_verdict(good) == "calibrated" and validation.calibration_verdict(bad) == "overconfident"
    assert validation.calibration_verdict(validation.reliability(obs, y, np.full(50, 0.1))) == "underconfident"


def test_blind_prediction_lock_verify_score_and_tamper(tmp_path):
    store = validation.BlindStore(tmp_path / "blind.sqlite3")
    sc = Scenario(al_mass_g=1.0, mode="open", duration_s=900.0, dim_um=20.0, c_naoh_M=2.0)
    s = lf.simulate_fast(lf.build_constants(sc, TRUTH), sc, 2.0)
    d = store.lock("blind1", sc, TRUTH, s["t"], s["gen"], 0.02 * s["gen"] + 1e-4)
    assert len(d) == 64 and store.verify("blind1")
    with pytest.raises(ValueError):
        store.lock("blind1", sc, TRUTH, s["t"], s["gen"], 0.02 * s["gen"])
    t_obs = np.linspace(60, 900, 15)
    y_obs = np.interp(t_obs, s["t"], s["gen"]) * 1.01
    res = store.score("blind1", t_obs, y_obs, np.full(15, 2e-4))
    assert res["coverage95"] > 0.8 and res["metrics"]["rmse"] > 0
    store.con.execute("UPDATE blind SET payload=replace(payload, '0.0', '0.1') WHERE id='blind1'")
    assert store.verify("blind1") is False
    with pytest.raises(ValueError):
        store.score("blind1", t_obs, y_obs, np.full(15, 2e-4))


def test_held_out_scoring_accuracy_claim_and_report(calib, tmp_path):
    exps, pred, fit = calib
    rng = np.random.default_rng(21)
    heldout = make_exps(rng, conds=((1.5, 28, 25), (2.5, 22, 30)))
    sc = validation.score(pred, pred.base.with_(**fit.theta), heldout)
    assert sc["verdict"] == "calibrated" and 0.85 < sc["coverage95"] <= 1.0
    claim = validation.accuracy_claim(sc)
    assert "Held-out RMSE" in claim and "No accuracy claim" in validation.accuracy_claim(None)
    html = validation.validation_report("test", sc, {"note": "x"}, tmp_path / "r.html")
    assert "<table>" in html and (tmp_path / "r.html").exists()
    over = validation.score(pred, pred.base.with_(**{**fit.theta, "Ea": 62e3}), heldout)
    assert over["verdict"] == "overconfident" and over["rmse_all"] > sc["rmse_all"]


def test_extrapolation_degrades_with_distance():
    rng = np.random.default_rng(31)
    conds = ((1.0, 25, 20), (2.0, 25, 20), (3.0, 25, 20), (4.0, 25, 20), (6.0, 25, 20))
    exps = make_exps(rng, conds=conds)
    out = validation.extrapolation_test(lambda es: Predictor(es), exps, "c_naoh_M", (1.0, 3.0), NAMES, fit_lsq)
    assert out["rows"][0]["distance"] < out["rows"][-1]["distance"] and out["n_calibration"] == 3
    assert all(np.isfinite(r["rms_norm_resid"]) for r in out["rows"])


# ---------------------------------------------------------------- benchmarks
def _dataset(tmp_path, name="syn1", **cond):
    sc = Scenario(al_mass_g=0.5, mode="open", duration_s=1500.0, dim_um=20.0, c_naoh_M=2.0, **cond)
    s = lf.simulate_fast(lf.build_constants(sc, ParamSet()), sc, 2.0)
    t = np.linspace(100, 1500, 15)
    d = {"id": name, "citation": "SYNTHETIC test fixture (not literature data)", "t_s": t.tolist(),
         "h2_mol": np.interp(t, s["t"], s["gen"]).tolist(), "digitization_sd_mol": 2e-4,
         "conditions": {"al_mass_g": 0.5, "dim_um": 20.0, "c_naoh_M": 2.0, "T0_C": 25.0, **{k: v for k, v in cond.items()}}}
    p = tmp_path / f"{name}.json"
    p.write_text(json.dumps(d))
    return p


def test_benchmark_loader_validates(tmp_path):
    p = _dataset(tmp_path)
    assert benchmarks.load_dataset(p)["id"] == "syn1"
    bad = json.loads(p.read_text())
    bad["citation"] = "x"
    (tmp_path / "bad.json").write_text(json.dumps(bad))
    with pytest.raises(ValueError):
        benchmarks.load_dataset(tmp_path / "bad.json")
    bad2 = json.loads(p.read_text())
    del bad2["t_s"]
    (tmp_path / "bad2.json").write_text(json.dumps(bad2))
    with pytest.raises(ValueError):
        benchmarks.load_dataset(tmp_path / "bad2.json")
    assert benchmarks.load_all(tmp_path / "nonexistent") == []


def test_plot_digitizer_roundtrip(tmp_path):
    from PIL import Image

    img = np.full((200, 300, 3), 255, np.uint8)
    xs = np.arange(20, 281)
    ys = (180 - 0.5 * (xs - 20)).astype(int)  # a descending line in pixel space = rising data
    img[ys, xs] = (255, 0, 0)
    Image.fromarray(img).save(tmp_path / "plot.png")
    pts = benchmarks.digitize_image(tmp_path / "plot.png", (255, 0, 0), tol=20)
    conv = benchmarks.calibrate_axes([(20, 180), (280, 50)], [(0.0, 0.0), (1000.0, 0.13)])
    x, y = conv(pts[:, 0], pts[:, 1])
    assert x[0] == pytest.approx(0.0, abs=5.0) and y[-1] == pytest.approx(0.13, abs=2e-3)
    assert np.all(np.diff(y) >= -1e-3)
    conv_log = benchmarks.calibrate_axes([(0, 0), (100, 100)], [(1.0, 1.0), (100.0, 100.0)], log_x=True, log_y=True)
    assert conv_log(np.array([50.0]), np.array([50.0]))[0][0] == pytest.approx(10.0)


def test_leaderboard_and_regression_gate(tmp_path):
    ds = [benchmarks.load_dataset(_dataset(tmp_path, "a")), benchmarks.load_dataset(_dataset(tmp_path, "b", T0_C=30.0, T_amb_C=30.0))]
    board = benchmarks.leaderboard(ds, levels=("reduced", "L1"))
    for lv in ("reduced", "L1"):
        agg = board["levels"][lv]["aggregate"]
        assert agg["rmse"] < 5e-4 and agg["coverage95"] > 0.8
    base = tmp_path / "baseline.json"
    base.write_text(json.dumps(board))
    assert benchmarks.regression_gate(board, base) == []
    worse = benchmarks.leaderboard(ds, ParamSet({"k25": 3e-4}), levels=("reduced",))
    assert benchmarks.regression_gate(worse, base)  # a worse model fails the gate
