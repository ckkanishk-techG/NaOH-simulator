import time

import numpy as np
import pytest

from hydra.constants import MW_AL
from hydra.core.geometry import make_bins, psd_masses
from hydra.core.l1 import simulate
from hydra.core.params import ParamSet
from hydra.core.scenario import PSDSpec, Scenario
from hydra.transport.l2 import L2Extras
from hydra.transport.regime import regime_map

BASE = dict(al_mass_g=1.0, mode="open", duration_s=7200.0, c_naoh_M=4.0, form="powder", dim_um=100.0)


def test_psd_kinds_conserve_mass_and_area():
    for kind in ("mono", "lognormal", "rosin_rammler"):
        sc = Scenario(psd=PSDSpec(kind=kind, n_bins=30), **BASE)
        b = make_bins(sc)
        assert b.n0.sum() * MW_AL == pytest.approx(1.0e-3, rel=1e-12)
        assert b.mass_frac.sum() == pytest.approx(1.0)
    d, w = psd_masses(PSDSpec(kind="custom", custom_d_um=[50, 100], custom_mass_frac=[1, 3]), 100.0)
    assert w.tolist() == [0.25, 0.75]
    mono = make_bins(Scenario(**BASE))
    assert mono.a0.sum() == pytest.approx(6.0 / 100e-6 * 1e-3 / 2700.0, rel=1e-9)  # 6/d * V


def test_L2_one_class_reduces_to_L1_when_mass_transfer_is_fast():
    l1 = simulate(Scenario(**BASE))
    l2 = simulate(Scenario(fidelity="L2", **BASE), ParamSet({"sh_min": 1.0e9}))
    n = len(l1.t)
    assert np.max(np.abs(l2["h2_gen_mol"][:n] - l1["h2_gen_mol"]) / l1["h2_gen_mol"].max()) < 1e-3


def test_sherwood_limits_and_agitation():
    sc = Scenario(fidelity="L2", **BASE)
    ex = L2Extras(ParamSet(), sc)
    d = np.array([100e-6])
    k0 = ex.kmt(298.15, 2.0, d, 0.0, None)[0]
    assert k0 == pytest.approx(2.0 * ex.diffusivity(298.15, 2.0) / 100e-6, rel=1e-9)  # Sh -> 2
    assert ex.kmt(298.15, 2.0, d, 1e-3, None)[0] > k0  # bubble-induced agitation
    assert ex.kmt(330.0, 2.0, d, 0.0, None)[0] > k0  # D rises with T
    assert ex.kmt(298.15, 2.0, np.array([50e-6]), 0.0, None)[0] > k0  # smaller d -> larger kmt
    assert Scenario(fidelity="L2", stirred=True, **BASE) and ex.kmt(298.15, 2.0, d, 0.0, None)[0] == k0
    st = L2Extras(ParamSet(), Scenario(fidelity="L2", stirred=True, **BASE))
    assert st.kmt(298.15, 2.0, d, 0.0, None)[0] > k0


def test_damkohler_regime_map_and_particle_size_trend():
    sc = Scenario(fidelity="L2", **BASE)
    big = regime_map(Scenario(fidelity="L2", **{**BASE, "dim_um": 400.0}))
    small = regime_map(sc)
    assert np.all(big["Da"] > small["Da"])  # larger particles -> smaller kmt -> larger Da
    assert set(np.unique(big["regime"])) <= {0, 1, 2}
    assert np.all(np.diff(small["Da"], axis=0) > 0)  # Da rises with T (kinetics outpace diffusion)


def test_finer_particles_react_faster_and_sizes_shrink_monotonically():
    t90 = []
    for d in (60.0, 150.0):
        t90.append(simulate(Scenario(fidelity="L2", **{**BASE, "dim_um": d})).summary["t90_s"])
    assert t90[0] < t90[1]
    r = simulate(Scenario(fidelity="L2", psd=PSDSpec(kind="lognormal", n_bins=8), **BASE))
    sizes = r.model.particle_sizes(r)
    assert np.all(np.diff(sizes, axis=0) <= 1e-12) and sizes[0, 3] > sizes[-1, 3]


def test_L2_50_bins_performance_and_conservation():
    sc = Scenario(fidelity="L2", psd=PSDSpec(kind="lognormal", n_bins=50, sigma_g=1.4), **{**BASE, "duration_s": 3600.0})
    t = time.time()
    r = simulate(sc)
    assert time.time() - t < 10.0  # spec: L2 (50 bins) < 10 s on a laptop CPU
    assert all(abs(r.ledger[k]) < 1e-3 for k in ("Al", "Na", "O", "H", "energy"))


def test_bubble_blocking_bounded_and_slows_reaction():
    off = simulate(Scenario(fidelity="L2", **BASE))
    on = simulate(Scenario(fidelity="L2", bubble_blocking=True, **BASE), ParamSet({"bub_N_site": 5.0e5}))
    assert on["theta_b"].max() > 0.05 and on["theta_b"].max() <= 0.95 + 1e-9
    assert on.summary["t90_s"] > off.summary["t90_s"]
    assert all(abs(on.ledger[k]) < 1e-3 for k in ("Al", "Na", "O", "H"))
    assert on.summary["h2_total_stp_L"] <= 1.2461 * (1 + 1e-3)  # never exceeds stoichiometry


def test_precipitate_coverage_blocks_surface():
    sc = Scenario(fidelity="L2", bubble_blocking=True, **BASE)
    ex = L2Extras(ParamSet(), sc)
    assert ex.blocked(0.0, 0.0, 0.0, 0.01, None) == 0.0
    assert ex.blocked(0.0, 0.0, 1e-3, 0.01, None) > 0.0
    assert ex.blocked(0.0, 0.0, 1e3, 0.01, None) <= 0.95 + 1e-12


def test_electrochemical_with_mass_transfer_runs_and_conserves():
    r = simulate(Scenario(fidelity="L2", rate_model="electrochemical", **BASE))
    assert r.summary["h2_total_stp_L"] == pytest.approx(1.246, rel=5e-3)
    assert r.summary["peak_Da"] >= 0.0
