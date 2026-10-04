import numpy as np
import pytest

from hydra.constants import MW_NAOH
from hydra.core.params import ParamSet
from hydra.core.scenario import Scenario
from hydra.design import oed
from hydra.inference.calibrate import Predictor, fit_lsq
from hydra.inference.data import synthetic_experiment

TRUTH = ParamSet({"k25": 3.0e-4, "Ea": 52e3, "n_oh": 0.9, "tau_ind": 40.0})
NAMES = ["k25", "Ea", "n_oh", "tau_ind"]
CONS = oed.LabConstraints()


@pytest.fixture(scope="module")
def setup():
    rng = np.random.default_rng(1)
    exps = []
    for i, (c, t0, d) in enumerate([(1.0, 25, 20), (2.0, 25, 20)]):
        sc = Scenario(al_mass_g=1.0, mode="open", duration_s=1500.0, dim_um=d, c_naoh_M=c, T0_C=t0, T_amb_C=t0)
        exps.append(synthetic_experiment(f"e{i}", sc, TRUTH, rng))
    fit = fit_lsq(Predictor(exps), NAMES)
    cands = oed.generate_candidates(CONS, n=40, seed=3)
    ranked = oed.rank_candidates(list(cands), fit, CONS)
    return exps, fit, ranked


def test_candidates_respect_lab_constraints():
    cands = oed.generate_candidates(CONS, n=25, seed=1)
    assert len(cands) == 25
    for c in cands:
        s = c.scenario
        assert s.c_naoh_M <= CONS.c_naoh_max_M and CONS.T0_range_C[0] <= s.T0_C <= CONS.T0_range_C[1]
        assert s.form in CONS.forms and c.peak_T_C <= CONS.peak_T_limit_C and c.conversion >= CONS.min_conversion
        assert s.v_liq_mL in CONS.v_liq_choices_mL and s.duration_s <= CONS.max_duration_s
    tight = oed.LabConstraints(c_naoh_max_M=1.0, peak_T_limit_C=30.0)
    for c in oed.generate_candidates(tight, n=5, seed=2):
        assert c.scenario.c_naoh_M <= 1.0 and c.peak_T_C <= 30.0


def test_oed_picks_more_informative_experiments_than_random(setup):
    _, _, ranked = setup
    eig = np.array([c.eig_focus for c in ranked])
    assert eig[0] >= eig[1] >= eig[-1]
    best3, rand = eig[:3].mean(), eig.mean()
    assert best3 > 1.5 * rand  # the OED choice beats a random candidate on average
    assert eig[0] > 5 * eig[-1]


def test_realised_information_gain_best_beats_worst(setup):
    exps, fit, ranked = setup
    rng = np.random.default_rng(5)

    def posterior_logdet(extra):
        f2 = fit_lsq(Predictor(exps + [extra]), NAMES)
        return float(np.linalg.slogdet(f2.cov)[1])

    def add(c):
        sc = c.scenario.model_copy(update={"duration_s": min(c.scenario.duration_s, 1500.0)})
        return synthetic_experiment("new", sc, TRUTH, rng)

    base = float(np.linalg.slogdet(fit.cov)[1])
    gain_best = base - posterior_logdet(add(ranked[0]))
    gain_worst = base - posterior_logdet(add(ranked[-1]))
    assert gain_best > gain_worst and gain_best > 0.5


def test_linear_eig_agrees_with_nested_monte_carlo_in_ranking(setup):
    exps, fit, ranked = setup
    from hydra.inference.bayes import Posterior, sample

    pred = Predictor(exps)
    post = Posterior(pred, NAMES)
    ch = sample(post, post.start(fit), nsteps=200, burn=100, backend="stretch", seed=0)
    xs = ch.flat()
    picks = [ranked[0], ranked[len(ranked) // 2], ranked[-1]]
    mc = [oed.eig_mc(c.scenario, ParamSet(), fit.space, xs, CONS, n_outer=40, n_inner=80) for c in picks]
    lin = [c.eig for c in picks]
    assert np.argsort(mc).tolist() == np.argsort(lin).tolist()
    assert mc[0] > mc[-1]


def test_protocol_contains_quantities_bands_and_stop_conditions(setup):
    _, fit, ranked = setup
    p = oed.make_protocol(ranked[0], fit, CONS)
    sc = ranked[0].scenario
    assert p.quantities["naoh_g"] == pytest.approx(sc.c_naoh_M * sc.v_liq_mL / 1e3 * MW_NAOH * 1e3)
    e = p.expected
    assert np.all(np.array(e["n_lo"]) <= np.array(e["n_med"]) + 1e-12) and np.all(np.array(e["n_med"]) <= np.array(e["n_hi"]) + 1e-12)
    assert e["n_med"][-1] <= 1.5 * sc.al_mass_g / 26.9815385 * 1.001
    md = p.to_markdown()
    assert "## Stop conditions" in md and "boiling" in md.lower() and "NOT a safety certification" in md
    assert "<pre" in p.to_html()
    assert len(p.schedule_s) >= 10
