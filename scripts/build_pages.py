"""Build the static browser edition into ``site/`` (Cloudflare Pages): the real HYDRA engine runs in Pyodide (WebAssembly)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "hydra"
OUT = ROOT / "site"
MODULES = ["__init__", "constants", "core/__init__", "core/geometry", "core/l1", "core/params", "core/rates", "core/safety",
           "core/scenario", "thermo/__init__", "thermo/electrolyte", "thermo/gas", "thermo/pitzer", "thermo/propdb",
           "thermo/ranges", "thermo/solubility", "thermo/species", "thermo/water"]

GLUE = '''
import json, warnings
import numpy as np
warnings.simplefilter("ignore")
from hydra.core.scenario import Scenario
from hydra.core.l1 import simulate, SolverSettings
from hydra.core.safety import analyze, DISCLAIMER
KEYS = ["h2_stp_L", "flow", "T", "Tw", "P", "al_g", "c_oh", "c_al", "sf", "Da", "conversion", "room_pct"]
def run(scenario_json):
    sc = Scenario(**json.loads(scenario_json))
    res = simulate(sc, None, SolverSettings(dt_out=max(sc.duration_s / 600.0, 1.0)))
    saf = analyze(res)
    f = lambda v: float(v) if np.isfinite(v) else None
    return json.dumps({
        "t": [f(x) for x in res.t], "series": {k: [f(x) for x in res.series[k]] for k in KEYS if k in res.series},
        "summary": {k: f(v) for k, v in res.summary.items()}, "ledger": {k: f(v) for k, v in res.ledger.items()},
        "safety": {"level": saf.level, "flags": [{"level": x.level, "code": x.code, "message": x.message} for x in saf.flags],
                   "forecasts": {k: f(v) for k, v in saf.forecasts.items()}},
        "ranges": [str(x) for x in res.diagnostics.get("range_violations", [])][:20], "disclaimer": DISCLAIMER})
'''


def main() -> None:
    OUT.mkdir(exist_ok=True)
    files = {f"hydra/{m}.py": (SRC / f"{m}.py").read_text() for m in MODULES}
    files["hydra/data/properties.json"] = (SRC / "data" / "properties.json").read_text()
    files["glue.py"] = GLUE
    (OUT / "engine.json").write_text(json.dumps(files))
    shutil.copy(ROOT / "scripts" / "pages" / "index.html", OUT / "index.html")
    shutil.copy(ROOT / "scripts" / "pages" / "worker.js", OUT / "worker.js")
    print(f"site/ built: {len(files)} engine files, {(OUT / 'engine.json').stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
