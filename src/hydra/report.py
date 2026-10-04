"""Local report generation (HTML always; PDF via WeasyPrint if installed, else matplotlib): inputs, assumptions,
equations, results, calibrated parameters with intervals, model-vs-experiment validation, safety summary, limitations.
Competition-appendix ready; nothing is fetched from the network (plots are embedded as base64 PNG)."""

from __future__ import annotations

import base64
import html
import io
from datetime import UTC, datetime
from typing import Any

import numpy as np

from . import plots
from .core.safety import DISCLAIMER, SafetyReport
from .provenance import RunRecord
from .thermo import propdb

EQUATIONS = [
    ("Overall reaction", r"2Al + 2NaOH + 6H_2O \rightarrow 2NaAl(OH)_4 + 3H_2"),
    ("Surface rate (empirical)", r"j = k_{25}\,e^{-E_a/R(1/T-1/T_{25})}\,c_{OH}^{\,n}\,\psi\,(1-\theta_b)"),
    ("Mixed potential", r"i_a(E_{corr}) = |i_c(E_{corr})|,\quad \dot n_{Al} = i_{corr}A/3F"),
    ("Energy balance", r"\Big(\sum_i n_i c_{p,i}\Big)\dot T = -Q_{lw} - Q_{cool} - \sum_i h_i \dot n_i"),
    ("Hoop stress", r"\sigma = P_g r / t,\quad SF = \sigma_y(T_w)/\sigma"),
    ("Cell voltage", r"V = E_{rev} - \eta_a - \eta_c - i\,ASR - \eta_{conc}"),
    ("Stoichiometry", r"n_{H_2} = \tfrac{3}{2} n_{Al}\;(1\,g\ Al \rightarrow 0.0556\,mol \approx 1.246\,L\ STP)"),
]
LIMITATIONS = [
    "All kinetic, thermal and fuel-cell parameters are UNCALIBRATED PRIORS unless a calibration is attached; accuracy is only claimed where measured on held-out data.",
    "Pitzer parameters are valid near 25 C; the model flags (but does not hide) use at higher temperature.",
    "Boiling is treated with a regularised superheat term; foaming, mist formation and liquid carry-over are not resolved.",
    "The reduced (fast) model has no vessel pressure, valve or boiling physics; the full model has no mechanical failure model.",
    "Fuel-cell membrane/water/carbonation/ageing parameters are placeholders until fitted to your membrane.",
    "This is a predictive research model, NOT a safety certification.",
]


def _png_b64(fig: Any) -> str:
    return base64.b64encode(plots.render(fig, "png", 150)).decode()


def assumptions_table(categories: tuple[str, ...] = ("param", "vessel", "electrochem", "fuelcell")) -> list[dict[str, Any]]:
    rows = []
    for c in categories:
        rows += [r for r in propdb.table(c) if r["status"] != "exact"]
    return rows


def build_html(title: str, rec: RunRecord | None, res: Any, safety: SafetyReport | None = None, calibration: dict[str, Any] | None = None,
               validation: dict[str, Any] | None = None, system: Any = None, extra_notes: str = "") -> str:
    e = html.escape
    parts = [f"<h1>{e(title)}</h1><p><i>{e(DISCLAIMER)}</i></p>",
             f"<p>Generated {datetime.now(UTC).isoformat(timespec='seconds')} UTC by HYDRA (local, offline).</p>"]
    if rec is not None:
        parts.append("<h2>Inputs and provenance</h2><table>" + "".join(
            f"<tr><td>{e(k)}</td><td><code>{e(str(v))[:300]}</code></td></tr>" for k, v in
            {"run id": rec.run_id, "timestamp (UTC)": rec.created_utc, "HYDRA version": rec.hydra_version, "git": rec.git_hash,
             "environment hash": rec.env_hash, "seed": rec.seed, "fidelity": rec.fidelity, "solver": rec.solver}.items()) + "</table>")
        parts.append("<h3>Scenario</h3><pre>" + e(str(rec.parameters.get("scenario"))) + "</pre>")
    parts.append("<h2>Assumptions (constants: units, range, status, source placeholder)</h2><table><tr><th>name</th><th>value</th><th>unit</th>"
                 "<th>range</th><th>status</th><th>source</th></tr>" + "".join(
        f"<tr><td>{e(r['name'])}</td><td>{r['value']:.4g}</td><td>{e(r['unit'])}</td><td>{r['min']}–{r['max']}</td>"
        f"<td>{e(r['status'])}</td><td>{e(r['source'])}</td></tr>" for r in assumptions_table()) + "</table>")
    parts.append("<h2>Governing equations</h2>" + "".join(f"<p><b>{e(n)}</b>: <code>{e(q)}</code></p>" for n, q in EQUATIONS))
    if res is not None:
        s = res.summary
        parts.append("<h2>Results</h2><table>" + "".join(f"<tr><td>{e(k)}</td><td>{v:.5g}</td></tr>" for k, v in s.items()) + "</table>")
        for nm, fn in (("h2", plots.plot_cumulative_h2), ("temperature", plots.plot_temperature), ("pressure", plots.plot_pressure)):
            parts.append(f"<h3>{nm}</h3><img alt='{nm}' src='data:image/png;base64,{_png_b64(fn(res))}'/>")
        cons = {k: v for k, v in res.ledger.items()}
        parts.append("<h3>Conservation residuals</h3><pre>" + e(str({k: f"{v:.2e}" for k, v in cons.items()})) + "</pre>")
        if res.diagnostics.get("range_violations"):
            parts.append("<h3>Valid-range warnings</h3><ul>" + "".join(f"<li>{e(v)}</li>" for v in res.diagnostics["range_violations"]) + "</ul>")
    if system is not None:
        parts.append("<h2>System (stack, DC/DC, load)</h2><table>" + "".join(f"<tr><td>{e(k)}</td><td>{v:.5g}</td></tr>" for k, v in system.summary.items())
                     + "</table>" + "".join(f"<p>⚠ {e(w)}</p>" for w in system.warnings))
    if calibration:
        parts.append("<h2>Calibrated parameters with intervals</h2><table><tr><th>parameter</th><th>median</th><th>95 % lo</th><th>95 % hi</th></tr>" + "".join(
            f"<tr><td>{e(k)}</td><td>{v['median']:.5g}</td><td>{v['lo']:.5g}</td><td>{v['hi']:.5g}</td></tr>" for k, v in calibration.items()) + "</table>")
    else:
        parts.append("<h2>Calibrated parameters</h2><p><b>None attached: every parameter is an uncalibrated prior.</b></p>")
    if validation:
        parts.append("<h2>Model vs experiment</h2><pre>" + e(str({k: v for k, v in validation.items() if k != 'rows'})) + "</pre>")
    if safety is not None:
        parts.append("<h2>Safety summary</h2><ul>" + "".join(f"<li><b>{e(f.level.upper())}</b> {e(f.code)}: {e(f.message)}</li>" for f in safety.flags) + "</ul>")
    parts.append("<h2>Limitations</h2><ul>" + "".join(f"<li>{e(x)}</li>" for x in LIMITATIONS) + "</ul>")
    if extra_notes:
        parts.append(f"<h2>Notes</h2><p>{e(extra_notes)}</p>")
    css = "body{font:14px/1.5 system-ui,sans-serif;max-width:960px;margin:2em auto;padding:0 1em}td,th{border:1px solid #bbb;padding:2px 8px;font-size:12px}img{max-width:100%}"
    return f"<!doctype html><html><head><meta charset='utf-8'><title>{e(title)}</title><style>{css}</style></head><body>{''.join(parts)}</body></html>"


def build_pdf(title: str, html_doc: str, res: Any = None, safety: SafetyReport | None = None) -> bytes:
    """PDF bytes: WeasyPrint if available, else a matplotlib multi-page PDF (summary, plots, safety, limitations)."""
    try:  # pragma: no cover - optional dependency
        from weasyprint import HTML

        return HTML(string=html_doc).write_pdf()  # type: ignore[no-any-return]
    except Exception:  # noqa: BLE001
        pass
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        fig = plt.figure(figsize=(8.27, 11.69))
        lines = [title, "", DISCLAIMER[:90], ""]
        if res is not None:
            lines += [f"{k}: {v:.5g}" for k, v in res.summary.items()]
        if safety is not None:
            lines += ["", "Safety:"] + [f"[{f.level}] {f.code}: {f.message[:80]}" for f in safety.flags]
        lines += ["", "Limitations:"] + [f"- {x[:95]}" for x in LIMITATIONS]
        fig.text(0.05, 0.97, "\n".join(lines), va="top", fontsize=7.5, family="monospace")
        pdf.savefig(fig)
        plt.close(fig)
        if res is not None:
            for fn in (plots.plot_cumulative_h2, plots.plot_temperature, plots.plot_pressure):
                f = fn(res)
                pdf.savefig(f)
                plt.close(f)
    return buf.getvalue()


_ = np
