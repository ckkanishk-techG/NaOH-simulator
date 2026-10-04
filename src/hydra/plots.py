"""Publication-quality plots (matplotlib, offline). Light theme for exports, dark theme for the UI; every function
returns a ``Figure``; :func:`render` writes PNG (300 dpi default) or SVG."""

from __future__ import annotations

import io
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .constants import P_ATM  # noqa: E402

THEMES = {
    "light": {"axes.facecolor": "white", "figure.facecolor": "white", "axes.edgecolor": "#333", "text.color": "#111",
              "axes.labelcolor": "#111", "xtick.color": "#111", "ytick.color": "#111", "grid.color": "#ddd"},
    "dark": {"axes.facecolor": "#1b1e23", "figure.facecolor": "#14161a", "axes.edgecolor": "#aaa", "text.color": "#e8eaed",
             "axes.labelcolor": "#e8eaed", "xtick.color": "#e8eaed", "ytick.color": "#e8eaed", "grid.color": "#333"},
}
PALETTE = ("#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2", "#b279a2")


def _fig(theme: str, w: float = 6.5, h: float = 4.0) -> tuple[Figure, Any]:
    with plt.rc_context(THEMES[theme]):  # type: ignore[arg-type]
        fig, ax = plt.subplots(figsize=(w, h))
    ax.grid(True, alpha=0.5, lw=0.5)
    return fig, ax


def plot_cumulative_h2(res: Any, band: dict[str, np.ndarray] | None = None, theme: str = "light") -> Figure:
    """Cumulative H2 at STP and at vessel conditions, theoretical maximum and (optional) uncertainty band."""
    fig, ax = _fig(theme)
    t = res.t / 60.0
    s = res.series
    ax.plot(t, s["h2_stp_L"], color=PALETTE[0], label="H$_2$ (STP)")
    ax.plot(t, s["h2_actual_L"], color=PALETTE[1], ls="--", label="H$_2$ (vessel T, P)")
    ax.axhline(s["h2_max_mol"][0] * 22.414, color=PALETTE[3], ls=":", label="theoretical max (STP)")
    if band is not None:
        ax.fill_between(band["t"] / 60.0, band["lo"] * 22.414, band["hi"] * 22.414, color=PALETTE[0], alpha=0.25, label="95 % band")
    ax.set_xlabel("time [min]")
    ax.set_ylabel("cumulative H$_2$ [L]")
    ax.legend(frameon=False)
    return fig


def plot_flow(res: Any, theme: str = "light") -> Figure:
    fig, ax = _fig(theme)
    ax.plot(res.t / 60.0, res["flow"] * 22.414 * 60.0 * 1e3, color=PALETTE[0], label="generation")
    if "stack_H2" in res.series:
        ax.plot(res.t[1:] / 60.0, np.diff(res["stack_H2"]) / np.diff(res.t) * 22.414 * 60.0 * 1e3, color=PALETTE[2], label="to stack")
    ax.set_xlabel("time [min]")
    ax.set_ylabel("H$_2$ flow [mL/min STP]")
    ax.legend(frameon=False)
    return fig


def plot_temperature(res: Any, band: dict[str, np.ndarray] | None = None, theme: str = "light") -> Figure:
    fig, ax = _fig(theme)
    ax.plot(res.t / 60.0, res["T"] - 273.15, color=PALETTE[3], label="liquid")
    ax.plot(res.t / 60.0, res["Tw"] - 273.15, color=PALETTE[0], label="HDPE wall")
    if band is not None:
        ax.fill_between(band["t"] / 60.0, band["lo"] - 273.15, band["hi"] - 273.15, color=PALETTE[3], alpha=0.25)
    ax.set_xlabel("time [min]")
    ax.set_ylabel("temperature [°C]")
    ax.legend(frameon=False)
    return fig


def plot_pressure(res: Any, theme: str = "light") -> Figure:
    fig, ax = _fig(theme)
    ax.plot(res.t / 60.0, (res["P"] - P_ATM) / 1e5, color=PALETTE[0])
    ax.set_xlabel("time [min]")
    ax.set_ylabel("gauge pressure [bar]")
    return fig


def plot_species(res: Any, theme: str = "light") -> Figure:
    fig, ax = _fig(theme)
    ax.plot(res.t / 60.0, res["al_g"], color=PALETTE[0], label="Al remaining [g]")
    ax.plot(res.t / 60.0, res["c_oh"], color=PALETTE[1], label="[OH$^-$] [mol/L]")
    ax.plot(res.t / 60.0, res["c_al"], color=PALETTE[2], label="[Al(OH)$_4^-$] [mol/L]")
    ax.set_xlabel("time [min]")
    ax.legend(frameon=False)
    return fig


def plot_particles(res: Any, theme: str = "light") -> Figure:
    """Particle-size evolution of every size class (L2)."""
    fig, ax = _fig(theme)
    sizes = res.model.particle_sizes(res) * 1e6
    for k in range(sizes.shape[1]):
        ax.plot(res.t / 60.0, sizes[:, k], color=PALETTE[0], alpha=0.25 + 0.75 * k / max(sizes.shape[1] - 1, 1))
    ax.set_xlabel("time [min]")
    ax.set_ylabel("characteristic size [µm]")
    return fig


def plot_heatmap(field: np.ndarray, grid: Any, title: str = "", cmap: str = "inferno", theme: str = "light") -> Figure:
    """L3/L4 field (shape (nz, nr)) as a half-section heat map."""
    fig, ax = _fig(theme, 5.0, 5.0)
    im = ax.pcolormesh(grid.r_f * 1e3, np.linspace(0.0, grid.H, grid.nz + 1) * 1e3, field, cmap=cmap, shading="flat")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("r [mm]")
    ax.set_ylabel("z [mm]")
    ax.set_title(title)
    return fig


def plot_regime_map(m: dict[str, np.ndarray], theme: str = "light") -> Figure:
    fig, ax = _fig(theme)
    cs = ax.contourf(m["c"], m["T"] - 273.15, np.log10(m["Da"]), levels=15, cmap="viridis")
    ax.contour(m["c"], m["T"] - 273.15, np.log10(m["Da"]), levels=[-1.0, 1.0], colors="white")
    fig.colorbar(cs, label="log$_{10}$ Da")
    ax.set_xlabel("[NaOH] [mol/L]")
    ax.set_ylabel("T [°C]")
    return fig


def plot_corrosion(res: Any, theme: str = "light") -> Figure:
    fig, ax = _fig(theme)
    ax.plot(res.t / 60.0, res["E_corr"], color=PALETTE[0])
    ax.set_xlabel("time [min]")
    ax.set_ylabel("E$_{corr}$ [V vs SHE]")
    ax2 = ax.twinx()
    ax2.plot(res.t / 60.0, res["i_corr"], color=PALETTE[1])
    ax2.set_ylabel("i$_{corr}$ [A/m$^2$]")
    return fig


def plot_polarization(stack: Any, theme: str = "light", n: int = 60) -> Figure:
    """Cell polarization and stack power curves."""
    fig, ax = _fig(theme)
    c = stack.cells[0]
    il = stack.cell.limiting_current(c)
    ii = np.linspace(1.0, 0.98 * il, n)
    v = np.array([stack.cell.voltage(i, c)["V"] for i in ii])
    ax.plot(ii / 1e4, v, color=PALETTE[0], label="cell voltage")
    ax.set_xlabel("current density [A/cm$^2$]")
    ax.set_ylabel("cell voltage [V]")
    ax2 = ax.twinx()
    ax2.plot(ii / 1e4, v * ii / 1e4 * stack.cell.area * 1e4 * stack.spec.n_series, color=PALETTE[1])
    ax2.set_ylabel("stack power [W]")
    return fig


def plot_stack_power(sysres: Any, theme: str = "light") -> Figure:
    fig, ax = _fig(theme)
    s = sysres.series
    ax.plot(s["t"] / 60.0, s["P_stack"], color=PALETTE[0], label="stack")
    ax.plot(s["t"] / 60.0, s["P_out"], color=PALETTE[2], label="to load")
    ax.set_xlabel("time [min]")
    ax.set_ylabel("power [W]")
    ax.legend(frameon=False)
    return fig


def plot_sankey(sk: dict[str, Any], theme: str = "light") -> Figure:
    """Three-column energy Sankey (matplotlib patches; the same data feeds a Plotly sankey in a browser UI)."""
    fig, ax = _fig(theme, 8.0, 4.5)
    ax.grid(False)
    ax.axis("off")
    nodes, links = sk["nodes"], sk["links"]
    inflow = np.zeros(len(nodes))
    outflow = np.zeros(len(nodes))
    for l in links:
        outflow[l["source"]] += l["value"]
        inflow[l["target"]] += l["value"]
    level = {i: 0 for i in range(len(nodes))}
    for _ in range(len(nodes)):
        for l in links:
            level[l["target"]] = max(level[l["target"]], level[l["source"]] + 1)
    cols: dict[int, list[int]] = {}
    for i, lv in level.items():
        cols.setdefault(lv, []).append(i)
    tot = max(max(inflow.max(), outflow.max()), 1e-12)
    pos: dict[int, tuple[float, float, float]] = {}
    for lv, ids in cols.items():
        y = 0.0
        for i in ids:
            h = max(inflow[i], outflow[i]) / tot
            pos[i] = (lv * 2.0, y, h)
            y += h + 0.03
    for l in links:
        x0, y0, h0 = pos[l["source"]]
        x1, y1, _ = pos[l["target"]]
        w = l["value"] / tot
        ax.fill([x0 + 0.3, x1, x1, x0 + 0.3], [y0, y1, y1 + w, y0 + w], alpha=0.35, color=PALETTE[l["source"] % len(PALETTE)])
    for i, (x, y, h) in pos.items():
        ax.add_patch(plt.Rectangle((x, y), 0.3, max(h, 0.005), color="#555"))
        ax.text(x + 0.35, y + h / 2, nodes[i], fontsize=7, va="center")
    ax.set_xlim(-0.2, max(c for c in cols) * 2.0 + 3.0)
    ax.set_ylim(-0.05, max(p[1] + p[2] for p in pos.values()) + 0.05)
    return fig


def plot_pareto(front: list[dict[str, Any]], x: str, y: str, theme: str = "light") -> Figure:
    fig, ax = _fig(theme)
    ax.scatter([f["objectives"][x] for f in front], [f["objectives"][y] for f in front], color=PALETTE[0])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return fig


PLOTS = {"h2": plot_cumulative_h2, "flow": plot_flow, "temperature": plot_temperature, "pressure": plot_pressure,
         "species": plot_species, "particles": plot_particles, "corrosion": plot_corrosion}


def render(fig: Figure, fmt: str = "png", dpi: int = 300) -> bytes:
    """Serialise a figure as PNG (default 300 dpi) or SVG."""
    if fmt not in ("png", "svg", "pdf"):
        raise ValueError("fmt must be png, svg or pdf")
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()
