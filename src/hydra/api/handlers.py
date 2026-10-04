"""Job handlers: each takes validated parameters and returns a JSON-able dict. All physics lives elsewhere."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..core.params import ParamSet
from ..core.safety import analyze, screen
from ..inference import bayes
from ..inference.calibrate import Predictor, fisher_information, fit_lsq
from ..inference.data import Channel, Experiment
from ..inference.sensors import Sensor
from ..jobs import JobContext, register
from .schemas import (
    CalibrateRequest,
    ControlRequest,
    OEDRequest,
    OptimizeRequest,
    ScaleupRequest,
    SimulateRequest,
    SpatialRequest,
)

_STATE: dict[str, Any] = {}


def bind(state: Any) -> None:
    """Called once by the app so handlers can reach the ELN / run repository."""
    _STATE["app"] = state


def _app() -> Any:
    return _STATE["app"]


def build_experiments(req: CalibrateRequest) -> list[Experiment]:
    exps: list[Experiment] = []
    for e in req.experiments:
        chans = []
        for c in e.channels:
            sg = np.full(len(c.t), c.sigma) if isinstance(c.sigma, (int, float)) else np.asarray(c.sigma)
            if np.any(sg <= 0):
                raise ValueError(f"channel {c.kind} of {e.id} needs a positive sigma (sensor uncertainty is propagated into the fit)")
            sens = Sensor(tau_s=c.tau_s) if c.kind in ("T", "Tw") else None
            chans.append(Channel(c.kind, np.asarray(c.t), np.asarray(c.y), sg, sens))
        exps.append(Experiment(e.id, e.scenario, chans, e.batch, e.split))
    for eid in req.eln_experiments:
        exps.append(_app().eln.to_experiment(eid))
    if not exps:
        raise ValueError("no experiments given")
    return exps


@register("simulate")
def simulate_job(params: dict[str, Any], ctx: JobContext) -> dict[str, Any]:
    req = SimulateRequest.model_validate(params)
    from ..core.l1 import SolverSettings

    ctx.progress(0.05, "running")
    rec, res = _app().runs.run(req.scenario, req.overrides, SolverSettings(rtol=req.rtol, dt_out=req.dt_out))
    saf = analyze(res, ParamSet(req.overrides))
    return {"run_id": rec.run_id, "summary": res.summary, "ledger": res.ledger, "safety": {"level": saf.level, "flags": [f.__dict__ for f in saf.flags]},
            "range_violations": res.diagnostics.get("range_violations", [])}


@register("system")
def system_job(params: dict[str, Any], ctx: JobContext) -> dict[str, Any]:
    from ..system.model import SystemSpec, simulate_system
    from .schemas import SimulateRequest as SR

    req = SR.model_validate({k: params[k] for k in ("scenario", "overrides") if k in params})
    spec = SystemSpec.model_validate(params.get("spec", {}))
    ctx.progress(0.05, "coupled system run")
    r = simulate_system(req.scenario, spec, ParamSet(req.overrides))
    keep = ("t", "I", "V", "P_stack", "P_out", "T_stack", "lam", "soc", "phi")
    return {"summary": r.summary, "warnings": r.warnings, "sankey": r.sankey, "series": {k: r.series[k].tolist() for k in keep if k in r.series},
            "vessel_summary": r.vessel.summary}


@register("calibrate")
def calibrate_job(params: dict[str, Any], ctx: JobContext) -> dict[str, Any]:
    req = CalibrateRequest.model_validate(params)
    exps = build_experiments(req)
    pred = Predictor(exps, ParamSet(), req.model, req.dt)
    ctx.progress(0.1, "least squares")
    fit = fit_lsq(pred, req.parameters)
    fi = fisher_information(fit)
    out: dict[str, Any] = {"theta": fit.theta, "sd": fit.sd(), "chi2": fit.chi2, "red_chi2": fit.red_chi2, "aic": fit.aic, "bic": fit.bic,
                           "cv": fi["cv"], "unidentifiable": fi["unidentifiable"], "condition": fi["cond"], "n_points": fit.n}
    intervals = {n: {"median": v, "lo": v - 1.96 * fit.sd()[n], "hi": v + 1.96 * fit.sd()[n]} for n, v in fit.theta.items()}
    if req.method == "bayes":
        post = bayes.Posterior(pred, req.parameters)
        ctx.progress(0.3, "MCMC")
        ch = bayes.sample(post, post.start(fit), nsteps=req.n_steps, burn=req.n_steps // 2, backend="stretch")
        intervals = bayes.credible_intervals(ch)
        out["diagnostics"] = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in ch.diag.items()}
        out["ppc"] = bayes.ppc_coverage(ch, 40)
        out["waic"] = bayes.waic(ch, 60)
    out["intervals"] = intervals
    if req.record_in_eln:
        out["calibration_id"] = _app().eln.record_calibration([e.id for e in exps], fit.theta, intervals, {"model": req.model, "method": req.method})
    return out


@register("oed")
def oed_job(params: dict[str, Any], ctx: JobContext) -> dict[str, Any]:
    from ..design import oed

    req = OEDRequest.model_validate(params)
    exps = build_experiments(req.calibration)
    pred = Predictor(exps, ParamSet(), req.calibration.model, req.calibration.dt)
    fit = fit_lsq(pred, req.calibration.parameters)
    cons = oed.LabConstraints(**req.constraints)
    ctx.progress(0.3, "generating candidates")
    cands = oed.generate_candidates(cons, req.n_candidates, req.seed)
    ranked = oed.rank_candidates(cands, fit, cons)
    top = ranked[: req.top_k]
    ctx.progress(0.8, "protocol")
    protos = [oed.make_protocol(c, fit, cons) for c in top]
    return {"n_feasible_candidates": len(cands),
            "ranked": [{"eig": c.eig, "eig_focus": c.eig_focus, "scenario": c.scenario.model_dump(), "peak_T_C": c.peak_T_C} for c in top],
            "protocols_markdown": [p.to_markdown() for p in protos]}


@register("optimize")
def optimize_job(params: dict[str, Any], ctx: JobContext) -> dict[str, Any]:
    from ..optimize import design as D

    req = OptimizeRequest.model_validate(params)
    goal = D.Goal(**req.goal)
    ctx.progress(0.1, req.method)
    if req.method == "pareto":
        front = D.pareto_nsga2(goal, tuple(req.objectives), pop=30, gens=max(req.maxiter // 2, 5), seed=req.seed)  # type: ignore[arg-type]
        return {"front": [{"design": f["design"], "objectives": f["objectives"]} for f in front]}
    if req.method == "bo":
        res = D.optimize_bo(goal, n_iter=req.maxiter, seed=req.seed)
    elif req.method == "robust":
        rng = np.random.default_rng(req.seed)
        draws = [ParamSet({"k25": 3e-4 * float(np.exp(0.4 * rng.standard_normal())), "Ea": 45e3 + 4e3 * float(rng.standard_normal())}) for _ in range(6)]
        res = D.optimize_robust(goal, draws, maxiter=req.maxiter, seed=req.seed)
    else:
        res = D.optimize_de(goal, maxiter=req.maxiter, seed=req.seed)
    out = {"design": res.design, "cost": res.metrics["cost"], "feasible": res.metrics["feasible"], "violations": res.metrics["violations"],
           "peak_T_C": res.metrics["peak_T_C"], "h2_total_mL": res.metrics["h2_total_mL"], "n_eval": res.n_eval}
    if req.verify:
        v = D.verify_design(res, goal)
        out["verification_full_model"] = {k: x for k, x in v.items() if k != "result"}
    return out


@register("control")
def control_job(params: dict[str, Any], ctx: JobContext) -> dict[str, Any]:
    from ..control import mpc, pid, sim
    from ..control import plant as pl

    req = ControlRequest.model_validate(params)

    def mk(load: list[tuple[float, float]] | None = None) -> pl.DosingPlant:
        return pl.DosingPlant(pl.drive_scenario(load or req.load_steps, req.duration_s), dt=req.dt, u_max_mL_min=4.0)

    ctrls: dict[str, Any] = {}
    ctx.progress(0.1, "tuning")
    if "pid" in req.controllers:
        g, _ = pid.autotune_relay(mk([(0.0, 1.5)]), req.setpoint_bar_g, u_mid=0.3, d=0.3, duration_s=5000.0, hyst=0.01)
        ctrls["pid"] = pid.PID(g, req.setpoint_bar_g, 4.0, req.dt, u_bias=0.3)
    if "dmc" in req.controllers:
        s = mpc.identify_step_response(mk([(0.0, 1.5)]), 0.2, du=0.1, horizon_s=450.0, warmup_s=60.0)
        ctrls["dmc"] = mpc.DMC(s, req.setpoint_bar_g, 4.0, req.dt, np_h=min(80, len(s) - 1), nu=10, lam=0.003, u_ref=0.3)
    if "open_loop" in req.controllers:
        ctrls["open_loop"] = lambda t, o: 0.3
    ctx.progress(0.5, "closed-loop runs")
    res = sim.compare_controllers(mk, ctrls, req.duration_s, req.setpoint_bar_g)
    return {"metrics": sim.metrics_table(res), "series": {n: {k: v.tolist() for k, v in r.series.items() if k in ("t", "u", "P", "T")} for n, r in res.items()}}


@register("spatial")
def spatial_job(params: dict[str, Any], ctx: JobContext) -> dict[str, Any]:
    from ..transport import axisym

    req = SpatialRequest.model_validate(params)
    ctx.progress(0.1, req.level)
    if req.level == "L3":
        if req.gci:
            res, unc = axisym.with_gci(lambda nr, nz: axisym.simulate_l3(req.scenario, nr=nr, n_out=60), ((max(req.nr // 2, 4), 1), (req.nr, 1), (req.nr * 2, 1)))
        else:
            res, unc = axisym.simulate_l3(req.scenario, nr=req.nr), {}
    else:
        res = axisym.simulate_l4(req.scenario, nr=req.nr, nz=req.nz, settled_fraction=req.settled_fraction)
        unc = {}
    return {"summary": res.summary, "numerical_uncertainty": unc, "series": {k: v.tolist() for k, v in res.series.items()}, "t": res.t.tolist(),
            "fields": [{"t": f["t"], "T": f["T"].tolist(), "c": f["c"].tolist(), "f": f["f"].tolist()} for f in res.fields],
            "grid": {"nr": res.grid.nr, "nz": res.grid.nz, "R": res.grid.R, "H": res.grid.H}}


@register("sobol")
def sobol_job(params: dict[str, Any], ctx: JobContext) -> dict[str, Any]:
    from ..core.scenario import Scenario
    from ..inference import sensitivity

    sc = Scenario.model_validate(params["scenario"])
    names = params.get("parameters", ["k25", "Ea", "n_oh", "tau_ind"])
    ctx.progress(0.1, "sobol")
    r = sensitivity.sobol_indices(sc, names, n=int(params.get("n", 64)), qoi=params.get("qoi", "h2_mol_at_tref"))
    return {"S1": {k: float(v) for k, v in r["S1"].items()}, "ST": {k: float(v) for k, v in r["ST"].items()}, "n_eval": r["n_eval"],
            "tornado": sensitivity.tornado(sc, names)}


def scaleup_compute(req: ScaleupRequest) -> dict[str, Any]:
    from ..optimize.cost import Prices
    from ..scaleup import economics, lca, sizing

    p = ParamSet(req.overrides)
    v = sizing.VehicleSpec(range_km=req.range_km, speed_kmh=req.speed_kmh, grade_pct=req.grade_pct)
    s = sizing.size_system(v, p)
    pr = Prices(**{"currency": req.currency, **req.prices})
    e = economics.techno_economics(s, pr, p, req.form, v, req.naoh_recovery, currency=req.currency)
    l = lca.lca_compare(s, p, req.recycled_share, req.naoh_recovery, v=v)
    return {"label": sizing.LABEL, "sizing": s.as_dict(), "economics": e.__dict__, "lca": l.__dict__}


@register("scaleup")
def scaleup_job(params: dict[str, Any], ctx: JobContext) -> dict[str, Any]:
    return scaleup_compute(ScaleupRequest.model_validate(params))


def safety_screen(scenario: Any, overrides: dict[str, float] | None = None) -> dict[str, Any]:
    rep = screen(scenario, ParamSet(overrides or {}))
    return {"level": rep.level, "unsafe": rep.unsafe, "flags": [f.__dict__ for f in rep.flags], "disclaimer": rep.disclaimer}
