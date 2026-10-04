import math
from pathlib import Path

import numpy as np
import pytest

from hydra.core.scenario import Scenario
from hydra.hardware import sources, twin

SC = Scenario(al_mass_g=1.0, mode="open", duration_s=2400.0, dim_um=40.0, form="powder", c_naoh_M=2.0, v_liq_mL=200.0)
ROOT = Path(__file__).resolve().parents[1]


def run_twin(method, hidden, r_mem=1.0, seed=1, t_stop=None):
    src = sources.SimulatedSource(SC, hidden=hidden, dt=5.0, seed=seed, stack_current=2.0, r_mem_mult=r_mem)
    tw = twin.LiveTwin(SC, cfg=twin.TwinConfig(method=method))
    rng = np.random.default_rng(seed)
    for f in src:
        if t_stop and f.t > t_stop:
            break
        cum = float(np.interp(f.t, src.truth["t"], src.truth["gen"])) + 3e-4 * rng.standard_normal()
        tw.ingest(f, cum)
    return tw, src


def test_parse_line_handles_garbage_and_missing_channels():
    f = sources.parse_line("12.5,25.1,24.0,0.05,0.2,2.0,2.7")
    assert f and f.t == 12.5 and f.V == 2.7
    g = sources.parse_line("13,25.1,nan,0.05")
    assert g and math.isnan(g.T2) and math.isnan(g.V) and g.P == 0.05
    for bad in ("", "# comment", "t_s,T1_C,T2_C", "abc,1,2", "1", "nan,1,2"):
        assert sources.parse_line(bad) is None
    assert sources.parse_line(sources.format_line(f)).T1 == pytest.approx(25.1)


def test_csv_stream_serial_and_modbus_sources(tmp_path):
    p = tmp_path / "log.csv"
    p.write_text("t_s,T1_C\n1,25\n2,25.5\nbad\n3,26\n")
    frames = list(sources.CsvStreamSource(p, idle_s=0.0, sleep=lambda s: None))
    assert [f.t for f in frames] == [1.0, 2.0, 3.0]

    class FakeSerial:
        def __init__(self, *a, **k):
            self.lines = [b"t_s,T1_C\n", b"1,20\n", b"2,21\n", b""]
            self.written = []

        def readline(self):
            return self.lines.pop(0)

        def write(self, *a):  # must never be used
            self.written.append(a)

    ser = sources.SerialSource("COM1", opener=FakeSerial)
    assert [f.T1 for f in ser] == [20.0, 21.0] and ser.ser.written == []

    class FakeClient:
        def read_holding_registers(self, addr, n, slave=1):
            class R:
                registers = [addr * 10]

                def isError(self):
                    return addr == 5

            return R()

    mb = sources.ModbusSource(FakeClient(), {"T1": (1, 0.1, 0.0), "P": (5, 1.0, 0.0)})
    fr = mb.read()
    assert fr.T1 == pytest.approx(1.0) and math.isnan(fr.P)
    with pytest.raises(ValueError):
        sources.ModbusSource(FakeClient(), {"bogus": (1, 1, 0)})


def test_simulated_source_has_realistic_noise_lag_and_drift():
    src = sources.SimulatedSource(SC, dt=5.0, seed=3)
    fr = list(src)
    t1 = np.array([f.T1 for f in fr])
    true_T = np.interp([f.t for f in fr], src.truth["t"], src.truth["T"]) - 273.15
    assert np.std(t1 - true_T) > 0.05  # noise
    assert not np.allclose(t1, true_T, atol=0.3)  # lag/drift visible
    assert fr[0].P == 0.0 and math.isnan(fr[0].V)


def test_ekf_recovers_hidden_rate_constant_resistance_and_remaining_aluminium():
    tw, src = run_twin("ekf", {"k25": 3.2e-4}, r_mem=1.5, t_stop=1100.0)
    st = tw.status()
    assert st["k_mult_best"] == pytest.approx(1.6, rel=0.08)
    assert st["R_mem_mult"] == pytest.approx(1.5, rel=0.05)
    f_true = float(np.interp(tw.t, src.truth["t"], src.truth["f"]))
    assert st["al_left_frac"] == pytest.approx(f_true, abs=0.03)
    assert st["h2_mol"] == pytest.approx(float(np.interp(tw.t, src.truth["t"], src.truth["gen"])), abs=2e-3)


def test_enkf_also_tracks_and_uncertainty_shrinks():
    tw, src = run_twin("enkf", {"k25": 1.2e-4}, t_stop=900.0)
    st = tw.status()
    assert st["k_mult_best"] == pytest.approx(0.6, rel=0.2)
    assert tw.history[5]["k_mult_sd_ln"] > tw.history[-1]["k_mult_sd_ln"]
    assert abs(st["T_C"] - (float(np.interp(tw.t, src.truth["t"], src.truth["T"])) - 273.15)) < 0.5


def test_filter_beats_open_loop_prediction_and_handles_bad_frames():
    tw, src = run_twin("ekf", {"k25": 3.2e-4}, t_stop=600.0)
    open_loop = twin.LiveTwin(SC, cfg=twin.TwinConfig(method="ekf"))
    for _ in range(int(600 / 5)):
        open_loop.x = open_loop._advance(open_loop.x, 5.0)
    g_true = float(np.interp(600.0, src.truth["t"], src.truth["gen"]))
    assert abs(tw.status()["h2_mol"] - g_true) < 0.2 * abs(open_loop.x[twin.I_G] - g_true) + 1e-4
    t0 = tw.t
    tw.ingest(sources.Frame(t0 - 10.0, T1=999.0))  # out-of-order frame ignored
    assert tw.t == t0
    tw.ingest(sources.Frame(t0 + 5.0))  # no channels: predict only
    assert tw.t == t0 + 5.0 and np.isfinite(tw.status()["T_C"])


def test_forecast_bands_cover_future_truth_and_early_warning_fires():
    tw, src = run_twin("ekf", {"k25": 3.2e-4}, t_stop=900.0)
    fc = tw.forecast(600.0, step_s=10.0, n=80)
    t_true = np.interp(fc.t, src.truth["t"], src.truth["T"]) - 273.15
    cover = np.mean((t_true >= fc.T_lo - 0.3) & (t_true <= fc.T_hi + 0.3))
    assert cover > 0.8 and np.all(fc.T_lo <= fc.T_hi + 1e-9) and np.all(np.diff(fc.gen_med) >= -1e-9)
    tight = twin.TwinConfig(method="ekf", limit_T_C=33.0)
    tw2 = twin.LiveTwin(SC, cfg=tight)
    src2 = sources.SimulatedSource(SC, hidden={"k25": 3.2e-4}, dt=5.0, seed=1)
    for f in src2:
        if f.t > 600.0:
            break
        tw2.ingest(f, float(np.interp(f.t, src2.truth["t"], src2.truth["gen"])))
    w = tw2.warnings(600.0)
    assert any(x.code == "T_LIMIT" for x in w)


def test_ingest_speed_is_realtime():
    import time

    tw = twin.LiveTwin(SC, cfg=twin.TwinConfig(method="ekf"))
    fr = sources.Frame(5.0, T1=25.0, T2=25.0, I=2.0, V=2.7)
    t0 = time.perf_counter()
    for k in range(1, 21):
        tw.ingest(sources.Frame(5.0 * k, T1=25.0, T2=25.0, I=2.0, V=2.7), 0.0)
    assert (time.perf_counter() - t0) / 20 < 0.1 and fr.t == 5.0


def test_firmware_is_read_only_and_matches_the_csv_contract():
    ino = (ROOT / "src/hydra/hardware/firmware/hydra_logger/hydra_logger.ino").read_text()
    assert "t_s,T1_C,T2_C,P_bar_g,flow_L_min,I_A,V_V" in ino
    for forbidden in ("digitalWrite", "analogWrite", "ledcWrite", "pinMode(RELAY", "servo"):
        assert forbidden not in ino
    assert ",".join(sources.FIELDS) == "t_s,T1_C,T2_C,P_bar_g,flow_L_min,I_A,V_V"
    assert "READ-ONLY" in ino and "relief" in ino.lower()
    doc = (ROOT / "src/hydra/docs/HARDWARE.md").read_text()
    assert "emergency stop" in doc and "DS18B20" in doc
