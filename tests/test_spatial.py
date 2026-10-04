import numpy as np
import pytest

from hydra.core import l1_fast as lf
from hydra.core.scenario import Scenario
from hydra.transport import axisym, fvm
from hydra.transport.openfoam import export_case

SC = Scenario(al_mass_g=1.0, mode="open", duration_s=1500.0, dim_um=60.0, form="powder", c_naoh_M=2.0, v_liq_mL=200.0,
              T0_C=25.0, T_amb_C=25.0)


def test_mms_spatial_second_order_and_temporal_orders():
    m = fvm.mms_spatial_order((8, 16, 32, 64))
    assert np.all(np.abs(m["order"] - 2.0) < 0.1) and np.all(np.diff(m["err"]) < 0)  # theoretical order of convergence
    assert np.all(np.abs(fvm.theta_temporal_order(0.5)[1:] - 2.0) < 0.1)  # Crank-Nicolson
    assert np.all(np.abs(fvm.theta_temporal_order(1.0) - 1.0) < 0.1)  # backward Euler
    m2 = fvm.mms_spatial_order((8, 16, 32), nz=3, theta=0.5)  # same operator with axial cells (z-invariant solution)
    assert np.all(np.abs(m2["order"] - 2.0) < 0.15)


def test_gci_and_richardson():
    out = fvm.gci(1.0, 1.1, 1.125)  # errors shrink by 4x => p = 2
    assert out["p"] == pytest.approx(2.0, rel=1e-9) and out["f_extrapolated"] == pytest.approx(1.1333333333)
    assert out["gci_fine"] == pytest.approx(1.25 * (0.025 / 1.125) / 3.0)
    osc = fvm.gci(1.0, 1.2, 1.1)
    assert osc["monotone"] == 0.0 and osc["gci_fine"] > 0
    assert fvm.gci(2.0, 2.0, 2.0)["gci_fine"] == 0.0


def test_diffusion_operator_conserves_and_is_symmetric_positive():
    g = fvm.Grid(0.04, 0.05, 8, 6)
    a, b = fvm.diffusion_operator(g, np.full(g.n, 0.6))
    assert abs(a.sum(axis=0)).max() < 1e-10 and np.allclose(a.toarray(), a.toarray().T)  # no Robin: pure Neumann, symmetric
    ev = np.linalg.eigvalsh(a.toarray())
    assert ev.max() < 1e-10
    a2, b2 = fvm.diffusion_operator(g, np.full(g.n, 0.6), h_wall=10.0, phi_inf=300.0)
    steady = np.linalg.solve(a2.toarray(), -b2)
    assert np.allclose(steady, 300.0, atol=1e-8)  # steady state equals the Robin reference value


def test_L3_with_high_conductivity_equals_well_mixed_reduced_model():
    r = axisym.simulate_l3(SC, nr=12, conv_mult=1e6, bottom_loss=True)
    fast = lf.simulate_fast(lf.build_constants(SC, dt=1.0), SC, dt=1.0)
    i = np.searchsorted(fast["t"], r.t)
    assert np.max(np.abs(r.series["gen"] - fast["gen"][i])) / fast["gen"][-1] < 1e-3
    assert np.max(np.abs(r.series["T_mean"] - fast["T"][i])) < 0.05
    assert r.summary["max_T_spread_K"] < 1e-3


def test_L4_without_gradients_equals_L3():
    sc = SC.model_copy(update={"al_mass_g": 2.0, "duration_s": 1000.0})
    r3 = axisym.simulate_l3(sc, nr=8)
    r4 = axisym.AxisymModel(sc, nr=8, nz=6, settled_fraction=0.0, bottom_loss=False).run()
    assert np.max(np.abs(r4.series["gen"] - r3.series["gen"])) / r3.series["gen"][-1] < 1e-4
    assert np.max(np.abs(r4.series["T_mean"] - r3.series["T_mean"])) < 1e-3
    fld = r4.fields[-1]["T"]
    assert np.ptp(fld, axis=0).max() < 1e-4  # no axial gradient


def test_settled_bed_creates_hot_spot_and_conserves_aluminium():
    sc = SC.model_copy(update={"al_mass_g": 2.0, "duration_s": 1200.0})
    r4 = axisym.simulate_l4(sc, nr=8, nz=10)
    t_last = r4.fields[2]["T"]
    assert t_last[0].mean() > t_last[-1].mean() + 0.5  # bottom (bed) hotter than the top
    assert r4.summary["max_T_spread_K"] > 1.0 and r4.summary["bed_height_m"] > 0
    f = r4.fields[-1]["f"]
    n_left = (f * axisym.AxisymModel(sc, nr=8, nz=10, settled_fraction=1.0).n0_c.reshape(10, 8)).sum()
    n_tot = 2.0e-3 / 26.9815385e-3
    gen = r4.series["gen"][-1]
    assert gen == pytest.approx(1.5 * (n_tot - n_left), rel=2e-3)  # H2 produced = 1.5 x Al consumed


def test_natural_convection_reduces_gradients():
    sc = SC.model_copy(update={"al_mass_g": 2.0, "duration_s": 600.0, "dim_um": 40.0})
    with_c = axisym.AxisymModel(sc, nr=8, nz=1, convection=True).run()
    without = axisym.AxisymModel(sc, nr=8, nz=1, convection=False).run()
    assert with_c.summary["max_T_spread_K"] < without.summary["max_T_spread_K"]


def test_gci_report_on_spatial_outputs():
    sc = SC.model_copy(update={"al_mass_g": 1.0, "duration_s": 900.0})
    res, unc = axisym.with_gci(lambda nr, nz: axisym.simulate_l3(sc, nr=nr, n_out=60), ((4, 1), (8, 1), (16, 1)),
                               keys=("h2_total_mol", "peak_T_mean_K"))
    assert res.numerical_uncertainty is unc and set(unc) == {"h2_total_mol", "peak_T_mean_K"}
    assert all(np.isfinite(v["gci_fine"]) for v in unc.values()) and unc["h2_total_mol"]["gci_fine"] < 0.05


def test_openfoam_export(tmp_path):
    out = export_case(SC, tmp_path / "case")
    for f in ("system/blockMeshDict", "system/controlDict", "constant/transportProperties", "0/T", "0/U", "README.txt"):
        assert (out / f).exists()
    txt = (out / "0/T").read_text()
    assert "externalWallHeatFluxTemperature" in txt and "298.15" in txt
    assert "wedge" in (out / "system/blockMeshDict").read_text()
