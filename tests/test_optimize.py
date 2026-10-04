import numpy as np
import pytest

from hydra.core import l1_fast as lf
from hydra.core.params import ParamSet
from hydra.optimize import cost as C
from hydra.optimize import design as D

GOAL = D.Goal(flow_mL_min=40.0, duration_min=20.0, peak_T_C=60.0)


def test_cost_model_currency_and_components():
    from hydra.core.scenario import Scenario

    p = C.Prices()
    sc = Scenario(al_mass_g=10.0, form="foil", c_naoh_M=2.0, v_liq_mL=100.0, cooling_UA_W_K=1.0)
    cons = C.consumables_cost(sc, p)
    assert cons == pytest.approx(0.01 * p.al_foil_per_kg + 2.0 * 0.1 * 39.997e-3 * p.naoh_per_kg + 0.1 * p.water_per_L)
    assert C.total_cost(sc, p, n_cells=4, battery_Wh=10.0) > C.total_cost(sc, p)
    assert p.convert(100.0, "USD") == pytest.approx(100.0 * p.fx["USD"])
    assert C.al_price("can", p) < C.al_price("powder", p)


def test_evaluate_reports_constraints_and_flow_logic():
    x = np.array([np.log(0.6), 1, np.log(11.0), 1.7, 55.0, 0.0])
    r = D.evaluate(x, GOAL)
    assert r["feasible"] and r["h2_total_mL"] + GOAL.buffer_mL > GOAL.flow_mL_min * (GOAL.duration_min - GOAL.ramp_s / 60.0)
    # too little aluminium cannot meet the flow requirement
    r2 = D.evaluate(np.array([np.log(0.2), 1, np.log(11.0), 1.7, 55.0, 0.0]), GOAL)
    assert not r2["feasible"] and r2["violations"]["flow"] > 0
    # a hot, large charge violates the temperature limit
    r3 = D.evaluate(np.array([np.log(15.0), 1, np.log(11.0), 4.0, 100.0, 0.0]), GOAL)
    assert r3["violations"]["peak_T"] > 0
    assert D.relief_capacity_mol_s(ParamSet(), 1.5) > 1e-3


def test_differential_evolution_finds_cheap_feasible_design():
    res = D.optimize_de(GOAL, maxiter=30, popsize=10, seed=1)
    assert res.metrics["feasible"]
    rng = np.random.default_rng(0)
    b = np.array(D.Space().bounds(GOAL))
    feas_costs = []
    for _ in range(3000):
        x = b[:, 0] + (b[:, 1] - b[:, 0]) * rng.random(len(b))
        m = D.evaluate(x, GOAL)
        if m["feasible"]:
            feas_costs.append(m["cost"])
    assert len(feas_costs) > 0 and res.metrics["cost"] < np.min(feas_costs) * 1.05
    v = D.verify_design(res, GOAL)  # full model with the relief valve
    assert v["peak_P_bar_g"] <= GOAL.p_max_bar_g * 1.03 and v["min_sf"] >= GOAL.min_sf
    assert abs(v["peak_T_C"] - res.metrics["peak_T_C"]) < 6.0  # fast model is a faithful screen


def test_bayesian_optimization_reaches_comparable_cost_with_fewer_evaluations():
    de = D.optimize_de(GOAL, maxiter=30, popsize=10, seed=1)
    bo = D.optimize_bo(GOAL, n_init=20, n_iter=30, seed=0)
    assert bo.metrics["feasible"] and bo.n_eval < de.n_eval / 4
    assert bo.metrics["cost"] < 5.0 * de.metrics["cost"]  # within ~5x of the global optimum using ~40x fewer evaluations
    assert bo.history[-1] <= bo.history[0]


def test_robust_optimization_improves_feasibility_probability():
    rng = np.random.default_rng(2)
    draws = [ParamSet({"k25": 3.0e-4 * float(np.exp(0.5 * rng.standard_normal())), "Ea": 45e3 + 4e3 * float(rng.standard_normal())})
             for _ in range(8)]
    nominal = D.optimize_de(GOAL, maxiter=25, popsize=10, seed=3)
    robust = D.optimize_robust(GOAL, draws, maxiter=20, popsize=8, seed=3)
    p_nom = D.feasibility_probability(nominal.x, GOAL, C.Prices(), draws)
    p_rob = D.feasibility_probability(robust.x, GOAL, C.Prices(), draws)
    assert p_rob >= p_nom and p_rob >= 0.75
    assert robust.metrics["cost"] >= nominal.metrics["cost"] * 0.9  # robustness is not free


def test_pareto_front_is_nondominated_feasible_and_shows_tradeoff():
    front = D.pareto_nsga2(GOAL, ("cost", "peak_T"), pop=30, gens=15, seed=0)
    assert len(front) >= 3
    pts = np.array([[f["objectives"]["cost"], f["objectives"]["peak_T"]] for f in front])
    for i in range(len(pts)):
        for j in range(len(pts)):
            if i != j:
                assert not (np.all(pts[j] <= pts[i]) and np.any(pts[j] < pts[i]))
    order = np.argsort(pts[:, 0])
    assert pts[order[0], 1] >= pts[order[-1], 1]  # cheaper designs run hotter
    assert all(f["metrics"]["feasible"] for f in front)
    many = D.pareto_nsga2(GOAL, ("neg_yield", "cost", "mass", "volume", "peak_T"), pop=24, gens=8, seed=1)
    assert len(many) >= 1


def test_stack_and_battery_variables():
    g = D.Goal(flow_mL_min=40.0, duration_min=20.0, peak_T_C=60.0, include_stack=True, include_battery=True, battery_Wh_min=5.0)
    r = D.optimize_de(g, maxiter=30, popsize=10, seed=2)
    d = r.design
    per_cell = g.i_op_A_cm2 * g.cell_area_cm2 / (2 * 96485.33212)
    assert d["n_cells"] * per_cell >= 40.0 / 1e3 / 22.414 / 60.0 * 0.999 and d["battery_Wh"] >= 5.0 - 1e-9 and r.metrics["feasible"]


@pytest.mark.skipif(not lf.jax_available(), reason="JAX not installed")
def test_jax_gradient_matches_finite_differences_and_descends():
    g = D.Goal(flow_mL_min=40.0, duration_min=6.0, peak_T_C=60.0, ramp_s=60.0, buffer_mL=10.0)
    jp = D.JaxPenalty(g, "powder")
    x0 = np.array([np.log(0.3), np.log(15.0), 1.2, 60.0, 0.0])
    v, grad = jp.value_and_grad(x0)
    for i in range(3):
        h = 1e-4 * max(abs(x0[i]), 1.0)
        xp, xm = x0.copy(), x0.copy()
        xp[i] += h
        xm[i] -= h
        fd = (jp.value_and_grad(xp)[0] - jp.value_and_grad(xm)[0]) / (2 * h)
        assert grad[i] == pytest.approx(fd, rel=2e-2, abs=1e-1)
    res = D.optimize_gradient(g, "powder", x0, maxiter=15)
    assert res.history[-1] <= res.history[0]
