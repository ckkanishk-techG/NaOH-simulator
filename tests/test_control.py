import numpy as np
import pytest

from hydra.control import cycles, hil, mpc, pid, sim
from hydra.control import plant as pl

DT = 5.0
LOAD = [(0.0, 1.0), (400.0, 2.5), (900.0, 1.0), (1400.0, 3.0), (1900.0, 0.5)]
SP = 0.3


def make_plant(load=LOAD, **kw):
    return pl.DosingPlant(pl.drive_scenario(load, 2400.0, **kw), dt=DT, u_max_mL_min=4.0)


@pytest.fixture(scope="module")
def tuned():
    g, info = pid.autotune_relay(make_plant([(0.0, 1.5)]), SP, u_mid=0.3, d=0.3, duration_s=5000.0, hyst=0.01)
    s = mpc.identify_step_response(make_plant([(0.0, 1.5)]), 0.2, du=0.1, horizon_s=450.0, warmup_s=60.0)
    return g, info, s


def test_plant_step_is_consistent_and_snapshot_restores():
    p = make_plant()
    snap = p.snapshot()
    a = [p.step(0.4).P_bar_g for _ in range(20)]
    p.restore(snap)
    b = [p.step(0.4).P_bar_g for _ in range(20)]
    assert np.allclose(a, b) and p.liquid_volume_mL() > 40.0  # dosing adds liquid
    p.reset()
    assert p.total_dosed_mL == 0.0 and p.m.valve_open is False
    ob = p.step(100.0)  # command is clipped to the actuator range
    assert p.log[-1]["u"] == 4.0 and ob.sf > 1.0


def test_dosing_is_the_manipulated_variable_pressure_rises_with_dose():
    ps = []
    for u in (0.0, 0.3, 0.8):
        p = make_plant([(0.0, 0.0)])
        p.headspace = "h2"
        p.reset()
        for _ in range(60):
            p.step(u)
        ps.append(p.measure().P_bar_g)
    assert ps[0] < ps[1] < ps[2]


def test_relay_autotune_finds_ultimate_gain_and_period(tuned):
    g, info, _ = tuned
    assert info["ku"] > 0 and info["pu"] > 100.0 and len(info["switch_times"]) >= 5
    assert g.kp == pytest.approx(0.3125 * info["ku"]) and g.ti == pytest.approx(2.2 * info["pu"])
    z = pid.zn_gains(info["ku"], info["pu"], "ziegler_nichols")
    assert z.kp == pytest.approx(0.6 * info["ku"]) and z.td > 0


def test_fopdt_fit_and_imc_rule_on_a_known_response():
    t = np.linspace(0, 200, 201)
    k, tau, th = 2.0, 30.0, 8.0
    y = np.where(t > th, k * 0.5 * (1 - np.exp(-(t - th) / tau)), 0.0)
    kf, tf, thf = pid.fopdt_fit(t, y, 0.5)
    assert kf == pytest.approx(k, rel=1e-3) and tf == pytest.approx(tau, rel=1e-3) and thf == pytest.approx(th, abs=0.1)
    g = pid.imc_gains(kf, tf, thf)
    assert g.kp == pytest.approx(tau / (k * (max(th, 0.25 * tau) + th)), rel=1e-2) and g.ti == pytest.approx(tau, rel=1e-2)


def test_pid_anti_windup_and_limits():
    ctrl = pid.PID(pid.PIDGains(kp=5.0, ti=20.0), 0.3, 4.0, 1.0, u_bias=0.0)
    o = pl.Obs(0, 0.0, 25, 25, 40, 0.3, 1, 0, 0, 20, False)
    for _ in range(200):  # unreachable set-point: output saturates, the integrator must not wind up
        u = ctrl.update(0.0, o)
    assert u == 4.0 and ctrl.i_term < 4.0 * 3  # bounded integral state
    high = pl.Obs(0, 2.0, 25, 25, 40, 0.3, 1, 0, 0, 20, False)
    rec = [ctrl.update(0.0, high) for _ in range(30)]
    assert rec[-1] == 0.0 and rec[0] < 4.0  # recovers promptly (no wind-up lag): output drops
    rl = pid.PID(pid.PIDGains(kp=5.0, ti=float("inf")), 0.3, 4.0, 1.0, rate_limit=0.1)
    seq = [rl.update(0.0, o) for _ in range(5)]
    assert np.all(np.diff(seq) <= 0.1 + 1e-9)


def test_controllers_beat_open_loop_on_a_drive_cycle(tuned):
    g, _, s = tuned
    res = sim.compare_controllers(
        make_plant, {"open_loop": lambda t, o: 0.3, "pid": pid.PID(g, SP, 4.0, DT, u_bias=0.3),
                     "dmc": mpc.DMC(s, SP, 4.0, DT, np_h=80, nu=10, lam=0.003, u_ref=0.3)}, 2400.0, SP, 120.0)
    tab = {r["controller"]: r for r in sim.metrics_table(res)}
    assert tab["pid"]["iae_bar_s"] < 0.75 * tab["open_loop"]["iae_bar_s"]
    assert tab["dmc"]["iae_bar_s"] < 0.75 * tab["open_loop"]["iae_bar_s"]
    for r in tab.values():
        assert set(r) >= {"rms_pressure_error_bar", "starved_s", "peak_T_C", "min_sf", "safety_margin_T_C", "dosed_mL"}
        assert r["peak_P_bar_g"] <= 1.51 and r["min_sf"] > 3.0


def test_interlock_limits_temperature_and_pressure_even_for_a_reckless_controller():
    hot = lambda: pl.DosingPlant(pl.drive_scenario([(0.0, 0.0)], 1500.0, al_mass_g=15.0, dim_um=20.0, cooling_UA=0.0), dt=DT, u_max_mL_min=4.0)  # noqa: E731
    free = sim.run_closed_loop(hot(), lambda t, o: 4.0, 1500.0, SP, interlock=sim.Interlock(T_max_C=500.0, P_max_bar_g=99, sf_min=0, V_max_mL=1e9))
    safe = sim.run_closed_loop(hot(), lambda t, o: 4.0, 1500.0, SP, interlock=sim.Interlock(T_max_C=45.0, P_max_bar_g=1.0, V_max_mL=1e9))
    assert free.metrics["peak_T_C"] > 55.0
    # the interlock only stops DOSING: reactant already in the vessel still reacts, so it limits but cannot undo the excursion
    assert safe.metrics["interlock_trips"] > 0 and safe.interlock_trips
    assert safe.metrics["peak_T_C"] < free.metrics["peak_T_C"] - 5.0
    assert safe.metrics["dosed_mL"] < free.metrics["dosed_mL"]


def test_drive_cycles_and_csv_upload(tmp_path):
    assert cycles.step(0.5, 2.0, 100.0, 300.0) == [(0.0, 0.5), (100.0, 2.0)]
    r = cycles.ramp(0.0, 3.0, 100.0, 200.0, 6)
    assert r[-1][1] == pytest.approx(3.0) and all(a[1] <= b[1] for a, b in zip(r, r[1:]))
    sg = cycles.stop_go(600.0, seed=3)
    assert sg == cycles.stop_go(600.0, seed=3) and sg[0] == (0.0, 0.0) and max(i for _, i in sg) <= 3.0
    f = tmp_path / "c.csv"
    f.write_text("t_s,i_A\n0,0\n60,1.5\n120,3.0\n")
    assert cycles.load_csv(f) == [(0.0, 0.0), (60.0, 1.5), (120.0, 3.0)]
    (tmp_path / "bad.csv").write_text("t_s,i_A\n10,1\n5,2\n")
    with pytest.raises(ValueError):
        cycles.load_csv(tmp_path / "bad.csv")
    assert cycles.power_to_current([(0.0, 14.0)], 4, 0.7) == [(0.0, 5.0)]
    assert cycles.h2_demand_mol_s(2.0, 4) == pytest.approx(8.0 / (2 * 96485.33212))


def test_hil_is_read_only_by_default_and_actuation_needs_three_things():
    ctrl = pid.PID(pid.PIDGains(1.0, 100.0), SP, 4.0, 1.0)
    ob = pl.Obs(0, 0.1, 25, 25, 40, 0.3, 1, 0, 0, 20, False)
    r = hil.HILRunner(ctrl)
    out = r.step(ob)
    assert out["read_only"] and not out["written"] and out["proposed"] > 0
    sink = hil.NullSink()
    with pytest.raises(hil.ActuationRefused):  # a sink alone is not enough
        hil.HILRunner(ctrl, hil.HILConfig(actuation_enabled=False), sink=_RealSink())
    with pytest.raises(hil.ActuationRefused):  # no acknowledgement of the independent E-stop/relief
        hil.HILRunner(ctrl, hil.HILConfig(actuation_enabled=True), sim.Interlock(), sink=_RealSink())
    with pytest.raises(hil.ActuationRefused):  # no software interlock
        hil.HILRunner(ctrl, hil.HILConfig(actuation_enabled=True, ack_independent_estop_and_relief=True), None, sink=_RealSink())
    ok = hil.HILRunner(ctrl, hil.HILConfig(actuation_enabled=True, ack_independent_estop_and_relief=True), sim.Interlock(), sink=(rs := _RealSink()))
    res = ok.step(ob)
    assert res["written"] and rs.writes
    hot = pl.Obs(1, 0.1, 99.0, 99.0, 40, 0.3, 1, 0, 0, 20, False)  # interlock forces zero
    assert ok.step(hot)["filtered"] == 0.0 and rs.writes[-1] == 0.0
    assert "emergency stop" in hil.HARDWARE_WARNING and sink.log == []


class _RealSink(hil.ActuatorSink):
    def __init__(self):
        self.writes = []

    def write(self, u):
        self.writes.append(u)


def test_hil_watchdog_zeroes_output_on_stale_data():
    t = [0.0]
    ctrl = pid.PID(pid.PIDGains(1.0, 100.0), SP, 4.0, 1.0)
    rs = _RealSink()
    r = hil.HILRunner(ctrl, hil.HILConfig(actuation_enabled=True, ack_independent_estop_and_relief=True, watchdog_s=5.0),
                      sim.Interlock(), sink=rs, clock=lambda: t[0])
    r.last_data = 0.0
    t[0] = 3.0
    assert r.watchdog() is False
    t[0] = 20.0
    assert r.watchdog() is True and rs.writes[-1] == 0.0
