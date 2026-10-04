"""Export geometry + boundary conditions to an OpenFOAM case (optional; OpenFOAM is NOT required to run HYDRA).

Writes a 5-degree axisymmetric wedge case for ``buoyantFoam``-style natural-convection runs: blockMeshDict, field files
with the scenario's initial/ambient temperatures, wall heat-transfer boundary condition derived from the HDPE wall,
transport/thermophysical properties of the electrolyte, and a README with the assumptions. No solver is invoked.
"""

from __future__ import annotations

import math
from pathlib import Path

from ..core.params import ParamSet
from ..core.scenario import Scenario
from ..thermo import electrolyte
from ..thermo import propdb as DB

HEADER = "FoamFile\n{\n    version 2.0;\n    format ascii;\n    class %s;\n    object %s;\n}\n"


def _w(path: Path, cls: str, obj: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("/* HYDRA-generated OpenFOAM file */\n" + HEADER % (cls, obj) + "\n" + body + "\n")


def export_case(sc: Scenario, out_dir: str | Path, params: ParamSet | None = None, nr: int = 30, nz: int = 40) -> Path:
    p = params or ParamSet()
    out = Path(out_dir)
    R = p["r_ves"]
    vl = sc.v_liq_mL * 1e-6
    H = vl / (math.pi * R**2)
    wedge = math.radians(2.5)  # half-angle of the 5-degree wedge
    T0, Ta = sc.T0_C + 273.15, sc.T_amb_C + 273.15
    rho = electrolyte.density(sc.c_naoh_M, T0)
    mu = electrolyte.viscosity(sc.c_naoh_M, T0)
    h_wall = 1.0 / (1.0 / 8.0 + p["t_ves"] / p["k_hdpe"])  # magic: ~8 W/m2K outer natural convection, series with wall conduction
    y, x = R * math.sin(wedge), R * math.cos(wedge)
    _w(out / "system/blockMeshDict", "dictionary", "blockMeshDict", f"""convertToMeters 1;
vertices
(
    (0 0 0) ({x:.6g} {-y:.6g} 0) ({x:.6g} {y:.6g} 0) (0 0 0)
    (0 0 {H:.6g}) ({x:.6g} {-y:.6g} {H:.6g}) ({x:.6g} {y:.6g} {H:.6g}) (0 0 {H:.6g})
);
blocks ( hex (0 1 2 3 4 5 6 7) ({nr} 1 {nz}) simpleGrading (1 1 1) );
edges ();
boundary
(
    wall {{ type wall; faces ((1 2 6 5)); }}
    bottom {{ type wall; faces ((0 3 2 1)); }}
    top {{ type patch; faces ((4 5 6 7)); }}
    front {{ type wedge; faces ((0 4 7 3)); }}
    back {{ type wedge; faces ((2 6 5 1)); }}
    axis {{ type empty; faces (); }}
);""")
    _w(out / "system/controlDict", "dictionary", "controlDict", f"""application buoyantFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime {sc.duration_s:g};
deltaT 0.5; writeControl adjustableRunTime; writeInterval 30; adjustTimeStep yes; maxCo 0.5;""")
    _w(out / "system/fvSchemes", "dictionary", "fvSchemes", "ddtSchemes { default Euler; }\ngradSchemes { default Gauss linear; }\n"
       "divSchemes { default Gauss linearUpwind grad(U); }\nlaplacianSchemes { default Gauss linear corrected; }\n"
       "interpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }")
    _w(out / "system/fvSolution", "dictionary", "fvSolution", "solvers { \"(p|p_rgh|U|h|T)\" { solver PBiCGStab; preconditioner DILU; tolerance 1e-8; relTol 0.01; } }\n"
       "PIMPLE { nOuterCorrectors 1; nCorrectors 2; nNonOrthogonalCorrectors 0; pRefCell 0; pRefValue 0; }")
    _w(out / "constant/transportProperties", "dictionary", "transportProperties",
       f"// electrolyte {sc.c_naoh_M:g} M NaOH at {T0:.1f} K (HYDRA property correlations - uncalibrated)\n"
       f"rho {rho:.6g};\nmu {mu:.6g};\nkappa {p['k_liq']:.6g};\nCp {DB.get('cp_liq_mass'):.6g};\nbeta {p['rho_beta_T']:.6g};")
    _w(out / "constant/g", "uniformDimensionedVectorField", "g", "dimensions [0 1 -2 0 0 0 0];\nvalue (0 0 -9.80665);")
    _w(out / "0/T", "volScalarField", "T", f"""dimensions [0 0 0 1 0 0 0];
internalField uniform {T0:.6g};
boundaryField
{{
    wall {{ type externalWallHeatFluxTemperature; mode coefficient; h uniform {h_wall:.6g}; Ta uniform {Ta:.6g}; kappaMethod fluidThermo; value uniform {T0:.6g}; }}
    bottom {{ type externalWallHeatFluxTemperature; mode coefficient; h uniform {h_wall:.6g}; Ta uniform {Ta:.6g}; kappaMethod fluidThermo; value uniform {T0:.6g}; }}
    top {{ type zeroGradient; }}
    front {{ type wedge; }}
    back {{ type wedge; }}
}}""")
    _w(out / "0/U", "volVectorField", "U", "dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\n"
       "boundaryField { wall { type noSlip; } bottom { type noSlip; } top { type slip; } front { type wedge; } back { type wedge; } }")
    (out / "README.txt").write_text(
        "HYDRA OpenFOAM export\n=====================\n"
        f"Geometry: cylinder R = {R:.4f} m, liquid height H = {H:.4f} m (5 degree wedge). Fluid: {sc.c_naoh_M:g} M NaOH, rho = {rho:.1f} kg/m3.\n"
        "Reaction heat and hydrogen source terms are NOT included in these files: add a volumetric heat source "
        "fvOptions from the HYDRA L3/L4 reaction-rate field (see hydra.transport.axisym) if you want a coupled run.\n"
        "All properties come from uncalibrated HYDRA correlations - validate before use. This export needs no OpenFOAM to create.\n")
    return out
