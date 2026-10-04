"""Generate src/hydra/data/properties.json (the versioned property database).

Every entry: value (SI), unit, plausible range, 1-sigma uncertainty/prior, category, source, status.
status: 'prior'      = uncalibrated modelling prior (placeholder source) - NEVER treat as fact
        'unverified' = handbook-type value typed from memory; verify against your tables
        'exact'      = defined/standard number
Run:  python scripts/build_property_db.py
"""
import json
from pathlib import Path

DB = {}


def e(name, value, unit, lo, hi, unc=None, cat="param", src="[SRC-TODO]", status="prior",
      dist=None, sd=None, desc="", tmin=None, tmax=None):
    DB[name] = {"value": value, "unit": unit, "min": lo, "max": hi, "uncertainty": unc,
                "category": cat, "source": src, "status": status, "dist": dist, "sd": sd,
                "desc": desc, "T_min_K": tmin, "T_max_K": tmax}


# ---------------- kinetics (empirical / Arrhenius, L1) ----------------
e("k25", 2.0e-4, "mol m-2 s-1 (c=1 mol/L)", 5e-5, 1e-3, 1e-4, src="[SRC-K1] prototype prior", dist="logn", sd=0.6, desc="Surface dissolution rate constant at 25 C")
e("Ea", 45e3, "J mol-1", 35e3, 70e3, 6e3, src="[SRC-K2] prototype prior", dist="norm", sd=6e3, desc="Apparent activation energy")
e("n_oh", 0.8, "-", 0.5, 1.2, 0.1, src="[SRC-K3] prototype prior", dist="norm", sd=0.1, desc="Apparent reaction order in OH-")
e("tau_ind", 60.0, "s", 10.0, 600.0, None, src="[SRC-K4] prototype prior", dist="logn", sd=0.5, desc="Oxide-film induction time constant (empirical model)")
e("k_mt", 5.0e-5, "m s-1", 1e-5, 2e-4, None, src="[SRC-T1] prototype prior", dist="logn", sd=0.5, desc="OH- mass-transfer coefficient when Sherwood model is off")
e("mult_pure", 1.0, "-", 0.5, 1.5, None, src="reference", desc="Alloy rate multiplier: pure Al")
e("mult_1xxx", 0.95, "-", 0.5, 1.5, None, src="[SRC-A1]", dist="logn", sd=0.2, desc="Alloy rate multiplier: 1xxx")
e("mult_3xxx", 0.8, "-", 0.3, 1.5, None, src="[SRC-A2]", dist="logn", sd=0.3, desc="Alloy rate multiplier: 3xxx (cans)")
e("mult_6061", 0.7, "-", 0.3, 1.5, None, src="[SRC-A3]", dist="logn", sd=0.3, desc="Alloy rate multiplier: 6061")
e("can_active_coated", 0.6, "-", 0.1, 1.0, None, src="[SRC-A4]", desc="Active-area fraction of a crushed can with lacquer intact")
e("activator_gain_per_ppm", 0.0, "ppm-1", 0.0, 0.1, None, src="[SRC-A5] disabled by default", desc="Rate gain per ppm Ga/In/Sn activator (0 = feature off)")
e("rho_al", 2700.0, "kg m-3", 2650.0, 2720.0, 10.0, cat="material", src="[SRC-M1]", status="unverified", desc="Aluminium density")
e("rho_gib", 2420.0, "kg m-3", 2300.0, 2450.0, None, cat="material", src="[SRC-M2]", status="unverified", desc="Gibbsite density")
e("cp_al_mass", 897.0, "J kg-1 K-1", 880.0, 920.0, None, cat="material", src="[SRC-M3]", status="unverified", desc="Aluminium specific heat")

# ---------------- film / oxide (L1 empirical film) ----------------
e("film_stall_oh", 0.05, "mol L-1", 0.0, 0.5, None, src="[SRC-K5]", desc="OH- level where electrochemical film repassivation wins (used by electrochemical model only)")

# ---------------- species thermodynamic data at 298.15 K ----------------
for sp, hf, gf, s0, cp, src in [
    ("Al(s)", 0.0, 0.0, 28.3, 24.2, "NBS-type"),
    ("H2O(l)", -285.83e3, -237.13e3, 69.95, 75.30, "NBS-type"),
    ("H2O(g)", -241.83e3, -228.57e3, 188.83, 33.6, "NBS-type"),
    ("OH-(aq)", -230.0e3, -157.24e3, -10.75, -148.5, "NBS-type"),
    ("Na+(aq)", -240.12e3, -261.91e3, 59.0, 46.4, "NBS-type"),
    ("Al(OH)4-(aq)", -1502.5e3, -1305.3e3, 102.9, 50.0, "NBS-type; Cp is a placeholder (wide uncertainty)"),
    ("Al(OH)3(gibbsite)", -1293.1e3, -1154.9e3, 68.4, 93.1, "NBS-type"),
    ("H2(g)", 0.0, 0.0, 130.68, 28.84, "NBS-type"),
    ("H2(aq)", -4.2e3, 17.57e3, 57.7, 140.0, "approx.; Cp placeholder"),
    ("air(g)", 0.0, 0.0, 191.6, 29.1, "N2-like approx."),
]:
    for key, v, unit in (("hf", hf, "J mol-1"), ("gf", gf, "J mol-1"), ("s0", s0, "J mol-1 K-1"), ("cp", cp, "J mol-1 K-1")):
        wide = 0.3 if (sp.startswith("Al(OH)4") and key == "cp") else 0.0
        e(f"{key}:{sp}", v, unit, None, None, abs(v) * 0.002 + 50.0 if key in ("hf", "gf") else abs(v) * (0.05 + wide) + 1.0,
          cat="species", src=f"{src} [SRC-S-{sp}]", status="unverified",
          desc=f"{ {'hf':'Formation enthalpy','gf':'Gibbs energy of formation','s0':'Standard entropy','cp':'Heat capacity (taken constant)'}[key] } of {sp} at 298.15 K")
e("water_Tc", 647.096, "K", None, None, None, cat="species", src="IAPWS", status="unverified")
e("watson_exp", 0.38, "-", None, None, None, cat="species", src="Watson correlation", status="unverified")
e("hvap_ref", 44.0e3, "J mol-1", 40e3, 46e3, 100.0, cat="species", src="[SRC-S-H2O] steam tables", status="unverified", desc="Latent heat of water at 25 C (cross-check of hf H2O(g)-H2O(l))")

# ---------------- water vapour pressure (Antoine, mmHg, degC) ----------------
e("antoine_lo_A", 8.07131, "-", None, None, cat="species", src="NIST/Antoine 1-100 C", status="unverified", tmin=274.0, tmax=373.0)
e("antoine_lo_B", 1730.63, "-", None, None, cat="species", src="NIST/Antoine 1-100 C", status="unverified")
e("antoine_lo_C", 233.426, "-", None, None, cat="species", src="NIST/Antoine 1-100 C", status="unverified")
e("antoine_hi_A", 8.14019, "-", None, None, cat="species", src="NIST/Antoine 99-374 C", status="unverified", tmin=373.0, tmax=647.0)
e("antoine_hi_B", 1810.94, "-", None, None, cat="species", src="NIST/Antoine 99-374 C", status="unverified")
e("antoine_hi_C", 244.485, "-", None, None, cat="species", src="NIST/Antoine 99-374 C", status="unverified")

# ---------------- Pitzer (NaOH binary, 25 C) ----------------
e("pitzer_beta0_NaOH", 0.0864, "kg mol-1", None, None, 0.005, cat="electrolyte", src="Pitzer & Mayorga 1973 [SRC-P1]", status="unverified", tmin=288.0, tmax=308.0)
e("pitzer_beta1_NaOH", 0.253, "kg mol-1", None, None, 0.02, cat="electrolyte", src="Pitzer & Mayorga 1973 [SRC-P1]", status="unverified")
e("pitzer_cphi_NaOH", 0.0044, "kg2 mol-2", None, None, 0.002, cat="electrolyte", src="Pitzer & Mayorga 1973 [SRC-P1]", status="unverified")
e("pitzer_alpha", 2.0, "kg0.5 mol-0.5", None, None, None, cat="electrolyte", src="Pitzer convention", status="exact")
e("pitzer_b", 1.2, "kg0.5 mol-0.5", None, None, None, cat="electrolyte", src="Pitzer convention", status="exact")
e("debye_huckel_A_phi", 0.3915, "kg0.5 mol-0.5", None, None, 0.002, cat="electrolyte", src="Pitzer 1991 [SRC-P2]", status="unverified", desc="Debye-Huckel osmotic slope at 25 C")
e("pitzer_theta_OH_AlOH4", 0.0, "kg mol-1", -0.2, 0.2, None, cat="electrolyte", src="[SRC-P3] unknown, fit-able; 0 = ideal mixing", desc="Pitzer OH-/Al(OH)4- mixing term")
e("pitzer_beta0_NaAlOH4", 0.0, "kg mol-1", -0.2, 0.3, None, cat="electrolyte", src="[SRC-P3] unknown, fit-able", desc="Pitzer Na+/Al(OH)4- beta0")
e("pitzer_beta1_NaAlOH4", 0.0, "kg mol-1", -0.5, 0.8, None, cat="electrolyte", src="[SRC-P3] unknown, fit-able", desc="Pitzer Na+/Al(OH)4- beta1")

# ---------------- aluminate solubility / precipitation ----------------
e("sol_lnK0", -2.8, "ln(-)", -6.0, 0.0, None, cat="electrolyte", src="[SRC-C1] placeholder: calibrate to your solubility tables", dist="norm", sd=1.0, desc="ln K for Al(OH)3(s) + OH- = Al(OH)4-, K = a(Al(OH)4-)/a(OH-), at 25 C")
e("sol_dH", 40e3, "J mol-1", 10e3, 80e3, None, cat="electrolyte", src="[SRC-C2] placeholder", dist="norm", sd=10e3, desc="Van't Hoff enthalpy of the gibbsite dissolution equilibrium")
e("sol_polymorph_bayerite_factor", 1.6, "-", 1.0, 3.0, None, cat="electrolyte", src="[SRC-C3] placeholder", desc="Bayerite/gibbsite solubility ratio")
e("nuc_sigma", 0.03, "J m-2", 0.005, 0.1, None, cat="electrolyte", src="[SRC-C4] CNT placeholder", desc="Gibbsite-solution interfacial tension (CNT)")
e("nuc_A", 1.0e8, "m-3 s-1", 1e4, 1e12, None, cat="electrolyte", src="[SRC-C5] CNT placeholder", dist="logn", sd=2.0, desc="CNT pre-exponential factor")
e("growth_k", 1.0e-9, "m s-1", 1e-11, 1e-7, None, cat="electrolyte", src="[SRC-C6] placeholder", dist="logn", sd=1.0, desc="Linear growth-rate constant at unit relative supersaturation")
e("growth_order", 2.0, "-", 1.0, 3.0, None, cat="electrolyte", src="[SRC-C7] placeholder", desc="Growth-rate order in (S-1)")
e("vm_gibbsite", 6.4e-5, "m3 mol-1", 5e-5, 8e-5, None, cat="electrolyte", src="derived from rho_gib", status="unverified", desc="Molar volume of gibbsite")

# ---------------- gas / H2 ----------------
e("henry_H2", 7.8e-6, "mol m-3 Pa-1", 7e-6, 9e-6, 3e-7, cat="gas", src="[SRC-G1] Sander compilation", status="unverified", tmin=273.0, tmax=373.0, desc="Henry solubility of H2 in water at 25 C")
e("henry_H2_dlnk_d_invT", 500.0, "K", 300.0, 800.0, None, cat="gas", src="[SRC-G2] placeholder", desc="d ln kH / d(1/T)")
e("sechenov_H2_NaOH", 0.1, "L mol-1 (log10)", 0.05, 0.3, None, cat="gas", src="[SRC-G3] placeholder", dist="norm", sd=0.05, desc="Sechenov salting-out coefficient for H2 in NaOH")
e("kla_H2", 0.05, "s-1", 1e-3, 1.0, None, cat="gas", src="[SRC-G4] placeholder", dist="logn", sd=1.0, desc="Volumetric liquid->gas transfer coefficient for dissolved H2")
e("abel_noble_b", 1.55e-5, "m3 mol-1", 1.4e-5, 1.7e-5, None, cat="gas", src="[SRC-G5] Abel-Noble covolume for H2", status="unverified")
e("pr_Tc_H2", 33.19, "K", None, None, None, cat="gas", src="[SRC-G6] critical constants", status="unverified")
e("pr_Pc_H2", 1.313e6, "Pa", None, None, None, cat="gas", src="[SRC-G6] critical constants", status="unverified")
e("pr_omega_H2", -0.216, "-", None, None, None, cat="gas", src="[SRC-G6] critical constants", status="unverified")
e("cp_h2_gas", 28.84, "J mol-1 K-1", None, None, None, cat="gas", src="NBS-type", status="unverified")
e("evap_k", 0.01, "m s-1", 1e-3, 0.1, None, cat="gas", src="[SRC-G7] placeholder", dist="logn", sd=1.0, desc="Liquid-gas evaporation mass-transfer coefficient (water)")
e("boil_k", 0.05, "kg s-1 K-1", 0.005, 0.5, None, cat="gas", src="numerical: boiling superheat stiffness", desc="Boil-off per K of superheat (numerical regularisation)")

# ---------------- electrolyte property correlations (NaOH(aq)) ----------------
e("rho_water_25", 997.0, "kg m-3", None, None, 0.5, cat="electrolyte", src="[SRC-E0]", status="unverified")
e("rho_slope_c", 41.0, "kg m-3 per mol/L", 30.0, 50.0, 2.0, cat="electrolyte", src="[SRC-E1] fit to your density table", status="unverified", tmin=288.0, tmax=343.0, desc="d rho/dc for NaOH(aq), 25 C; valid c<=6 M")
e("rho_beta_T", 2.5e-4, "K-1", 1e-4, 5e-4, None, cat="electrolyte", src="[SRC-E2] approximate thermal expansion", status="unverified")
e("visc_water_A", 2.414e-5, "Pa s", None, None, None, cat="electrolyte", src="Vogel-type water viscosity", status="unverified")
e("visc_water_B", 247.8, "K", None, None, None, cat="electrolyte", src="Vogel-type water viscosity", status="unverified")
e("visc_water_C", 140.0, "K", None, None, None, cat="electrolyte", src="Vogel-type water viscosity", status="unverified")
e("visc_c1", 0.1, "L mol-1", None, None, None, cat="electrolyte", src="[SRC-E3] Jones-Dole-like, fit to table", status="prior")
e("visc_c2", 0.05, "L2 mol-2", None, None, None, cat="electrolyte", src="[SRC-E3] Jones-Dole-like, fit to table", status="prior")
e("cond_lambda0", 0.0095, "S m2 mol-1", None, None, None, cat="electrolyte", src="[SRC-E4] fit to table", status="prior", desc="Limiting-ish molar conductivity scale of NaOH(aq) at 25 C")
e("cond_k", 0.08, "L mol-1", None, None, None, cat="electrolyte", src="[SRC-E4] fit to table", status="prior")
e("cond_alpha_T", 0.02, "K-1", None, None, None, cat="electrolyte", src="[SRC-E4] approx temp coeff", status="prior")
e("sigma_water_25", 0.0720, "N m-1", None, None, None, cat="electrolyte", src="[SRC-E5]", status="unverified")
e("sigma_slope_T", -1.6e-4, "N m-1 K-1", None, None, None, cat="electrolyte", src="[SRC-E5]", status="unverified")
e("sigma_slope_c", 1.7e-3, "N m-1 per mol/L", None, None, None, cat="electrolyte", src="[SRC-E5] placeholder", status="prior")
e("k_liq", 0.60, "W m-1 K-1", 0.5, 0.7, None, cat="electrolyte", src="[SRC-E6]", status="unverified", desc="Liquid thermal conductivity (water-like)")
e("cp_liq_wall_dummy", 0.0, "-", None, None, None, cat="param", src="unused")
del DB["cp_liq_wall_dummy"]

# ---------------- vessel / heat transfer / HDPE ----------------
e("m_ves", 0.035, "kg", 0.025, 0.06, None, cat="vessel", src="measure", desc="HDPE vessel mass")
e("cp_hdpe", 1900.0, "J kg-1 K-1", 1800.0, 2300.0, None, cat="vessel", src="[SRC-V2]", status="prior")
e("k_hdpe", 0.45, "W m-1 K-1", 0.35, 0.5, None, cat="vessel", src="[SRC-V6]", status="prior")
e("emis_hdpe", 0.9, "-", 0.8, 0.95, None, cat="vessel", src="[SRC-V7]", status="prior")
e("r_ves", 0.04, "m", 0.03, 0.06, None, cat="vessel", src="measure", desc="Vessel inner radius")
e("t_ves", 0.0015, "m", 0.001, 0.003, None, cat="vessel", src="measure", desc="Wall thickness")
e("sy23", 26e6, "Pa", 20e6, 32e6, None, cat="vessel", src="[SRC-V3]", status="prior", desc="HDPE yield strength at 23 C")
e("dsy_dT", 0.25e6, "Pa K-1", 0.15e6, 0.35e6, None, cat="vessel", src="[SRC-V4]", status="prior")
e("T_hdpe_max", 353.15, "K", 333.15, 383.15, None, cat="vessel", src="[SRC-V5]", status="prior", desc="Max service temperature")
e("creep_A", 1.0e-9, "s-1", 1e-12, 1e-6, None, cat="vessel", src="[SRC-V8] placeholder", desc="Creep strain-rate prefactor (Norton)")
e("creep_n", 4.0, "-", 2.0, 8.0, None, cat="vessel", src="[SRC-V8] placeholder")
e("creep_Q", 60e3, "J mol-1", 20e3, 120e3, None, cat="vessel", src="[SRC-V8] placeholder")
e("creep_sref", 10e6, "Pa", None, None, None, cat="vessel", src="reference stress for creep law")
e("h_out_forced", 10.0, "W m-2 K-1", 2.0, 100.0, None, cat="vessel", src="[SRC-V9] placeholder", desc="Forced-convection external coefficient (fan/vehicle)")
e("valve_Av_per_Cv", 2.4e-5, "m2 per Cv", None, None, None, cat="vessel", src="unit conversion (Cv->orifice area)", status="unverified")
e("valve_Cv", 0.05, "Cv", 0.001, 2.0, None, cat="vessel", src="[SRC-V10] placeholder", desc="Relief-valve flow coefficient")
e("valve_hyst", 0.1, "-", 0.01, 0.5, None, cat="vessel", src="[SRC-V10] placeholder", desc="Reseat pressure = (1-hyst)*set pressure (gauge)")
e("valve_Cd", 0.65, "-", 0.5, 0.9, None, cat="vessel", src="[SRC-V11]", status="prior")
e("reg_dP_min", 5e3, "Pa", 0.0, 5e4, None, cat="vessel", src="[SRC-V12]", desc="Minimum gauge pressure for regulated outflow")
e("reg_width", 2e3, "Pa", 100.0, 2e4, None, cat="vessel", src="numerical smoothing width")
e("purge_flow", 3.0e-4, "mol s-1", 0.0, 1e-2, None, cat="vessel", src="[SRC-V13] anode purge bleed")
e("h2_min_frac", 0.9, "-", 0.5, 0.999, None, cat="vessel", src="[SRC-V14] min H2 fraction accepted by stack")
e("headroom_bubbles_v", 0.0, "m3", None, None, None, cat="vessel", src="unused")
del DB["headroom_bubbles_v"]

# ---------------- safety limits ----------------
e("lfl_h2", 0.04, "-", None, None, None, cat="safety", src="[SRC-S1] H2 in air flammable limits", status="unverified")
e("ufl_h2", 0.75, "-", None, None, None, cat="safety", src="[SRC-S1]", status="unverified")
e("safety_P_margin", 0.8, "-", 0.5, 1.0, None, cat="safety", src="engineering choice", desc="Warn when P_gauge > margin * relief set point/ burst estimate")
e("safety_sf_min", 3.0, "-", 1.5, 10.0, None, cat="safety", src="engineering choice", desc="Minimum acceptable hoop safety factor")
e("semenov_psi_crit", 1.0, "-", 0.5, 1.5, None, cat="safety", src="Semenov criterion", desc="Critical Semenov number (Psi)")
e("naoh_corrosive_M", 0.5, "mol L-1", None, None, None, cat="safety", src="[SRC-S2] ~pH 13.7 corrosive/irritant guidance")
e("room_volume", 30.0, "m3", 1.0, 1000.0, None, cat="safety", src="scenario default", desc="Default room volume")
e("room_ach", 0.5, "h-1", 0.0, 20.0, None, cat="safety", src="scenario default", desc="Default air changes per hour")
e("T_amb", 298.15, "K", 273.15, 323.15, None, cat="safety", src="scenario default")

if __name__ == "__main__":
    import sys

    out = Path(__file__).resolve().parents[1] / "src/hydra/data/properties.json"
    for f in sorted(Path(__file__).parent.glob("props_*.py")):  # later stages append entries here
        exec(compile(f.read_text(), str(f), "exec"), {"e": e, "DB": DB})
    out.write_text(json.dumps(DB, indent=1, sort_keys=True))
    print(f"wrote {len(DB)} entries to {out}", file=sys.stderr)
