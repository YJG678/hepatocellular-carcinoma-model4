"""Generate Figs. 8, 9 and 11 exclusively from regenerated result files."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import NullFormatter

from analysis_common import monotone_curve
from config import FIGURES, RESULTS
from plot_style import BLUE, INK, MUTE, ORANGE, TEAL, apply_style, clean_axis


STYLES = {
    "xi": (BLUE, "o", r"$\xi$: chemotactic sensitivity, $0\!-!24$"),
    "a": (ORANGE, "s", r"$a$: interior half-width, $2\!-!8$"),
    "d2": (TEAL, "^", r"$d_2$: effector motility, $0.01\!-!0.64$"),
}


def load_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def generate_figure9():
    rows = load_json(RESULTS / "sweeps.json")
    fit = load_json(RESULTS / "collapse_fit.json")
    access = np.array([row["A"] for row in rows])
    response = np.array([row["mean_int"] for row in rows])
    sweeps = np.array([row["sweep"] for row in rows])
    lhs = np.array([row["weighted_l1_lhs"] for row in rows])
    rhs = np.array([row["weighted_l1_rhs"] for row in rows])
    parameters = [fit["y0"], fit["log_Ac"], fit["scale"]]

    figure, axes = plt.subplots(1, 3, figsize=(11.4, 3.35))
    axis = axes[0]
    clean_axis(axis)
    low, high = 0.6 * min(lhs.min(), rhs.min()), 1.6 * rhs.max()
    axis.plot([low, high], [low, high], color=MUTE, lw=0.9, ls="--", zorder=1)
    axis.fill_between([low, high], [low, low], [low, high], color=MUTE, alpha=0.07, lw=0)
    for name, (colour, marker, label) in STYLES.items():
        selected = sweeps == name
        axis.scatter(
            rhs[selected],
            lhs[selected],
            s=26,
            marker=marker,
            facecolor=colour,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
            label=label,
        )
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlim(low, high)
    axis.set_ylim(low, high)
    axis.set_xlabel("analytic bound, right side")
    axis.set_ylabel(r"measured $\int_\Omega v e^{-\kappa_\varepsilon|x-x_0|}\,dx$")
    axis.text(0.05, 0.93, "A", transform=axis.transAxes, fontsize=11, fontweight="bold")
    axis.text(0.96, 0.90, f"27/27 runs\nmargin {fit['weighted_l1_ratio_min']:.1f}--{fit['weighted_l1_ratio_max']:.1f}$\\times$", transform=axis.transAxes, ha="right", va="top", fontsize=8.5, color=MUTE)

    axis = axes[1]
    clean_axis(axis)
    grid = np.linspace(np.log(access.min() * 0.8), np.log(access.max() * 1.2), 300)
    axis.plot(np.exp(grid), monotone_curve(grid, *parameters), color=MUTE, lw=1.1)
    axis.axvline(fit["Ac"], color=MUTE, lw=0.8, ls=":")
    for name, (colour, marker, label) in STYLES.items():
        selected = sweeps == name
        axis.scatter(access[selected], response[selected], s=30, marker=marker, facecolor=colour, edgecolor="white", linewidth=0.6, zorder=3, label=label)
    axis.set_xscale("log")
    axis.set_xlabel(r"immune access number $\mathcal{A}=1/(\kappa_*a)$")
    axis.set_ylabel(r"interior mean tumour density $\bar{u}_{\rm int}$")
    axis.set_ylim(0, 0.85)
    axis.set_xticks([0.15, 0.3, 0.5, 1.0, 1.5])
    axis.set_xticklabels(["0.15", "0.3", "0.5", "1.0", "1.5"])
    axis.text(0.05, 0.93, "B", transform=axis.transAxes, fontsize=11, fontweight="bold")
    axis.text(0.97, 0.93, rf"$R^2={fit['r2']:.6f}$" + "\n" + rf"RMSE $={fit['rmse']:.6f}$", transform=axis.transAxes, ha="right", fontsize=8.5)
    axis.legend(loc="lower left", fontsize=7.6, handletextpad=0.4, borderaxespad=0.2, bbox_to_anchor=(-0.02, -0.02))

    axis = axes[2]
    clean_axis(axis)
    candidates = [
        (r"$\mathcal{A}=1/(\kappa_*a)$", fit["r2"]),
        (r"$\xi$ alone", fit["null_fits"]["xi"]["r2_display"]),
        (r"$\kappa_*^{-1}$ alone", fit["null_fits"]["penetration_depth"]["r2_display"]),
        (r"$d_2$ alone", fit["null_fits"]["d2"]["r2_display"]),
        (r"$a$ alone", fit["null_fits"]["a"]["r2_display"]),
    ]
    positions = np.arange(len(candidates))[::-1]
    values = [value for _, value in candidates]
    axis.barh(positions, values, height=0.55, color=[BLUE] + [MUTE] * 4, edgecolor="none")
    for position, value in zip(positions, values):
        axis.text(max(value, 0.0) + 0.02, position, f"{value:.3f}", va="center", fontsize=8.5)
    axis.set_yticks(positions)
    axis.set_yticklabels([name for name, _ in candidates], fontsize=9)
    axis.set_xlim(0, 1.16)
    axis.set_xlabel(r"$R^2$ of one monotone fit to all 27 runs")
    axis.grid(axis="y", visible=False)
    axis.text(0.05, 0.93, "C", transform=axis.transAxes, fontsize=11, fontweight="bold")
    figure.tight_layout(w_pad=2.2)
    figure.savefig(FIGURES / "fig9_access_number.pdf")
    figure.savefig(FIGURES / "fig9_access_number.png", dpi=170)
    plt.close(figure)


def generate_figure8():
    data = np.load(RESULTS / "phase_grid.npz")
    stats = load_json(RESULTS / "phase_grid_stats.json")
    fit = load_json(RESULTS / "collapse_fit.json")
    sigma_hi, rm = data["sigma_hi"], data["rm"]
    mass, mean, access, supply = data["mass"], data["mean"], data["access"], data["supply"]
    parameters = [fit["y0"], fit["log_Ac"], fit["scale"]]
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.4, 4.0))
    clean_axis(left)
    X, Y = np.meshgrid(sigma_hi, rm)
    image = left.pcolormesh(X, Y, mass, cmap="PuBu", shading="gouraud", rasterized=True)
    colourbar = figure.colorbar(image, ax=left, pad=0.02)
    colourbar.set_label("interior refuge mass", fontsize=9)
    contour = left.contour(X, Y, supply, levels=[0.5], colors=[ORANGE], linewidths=1.7, linestyles="--")
    left.clabel(contour, fmt={0.5: r"$\overline{\sigma_0}=\delta$"}, fontsize=9)
    left.set_xlabel(r"margin recruitment level $\sigma_0^{\rm hi}$")
    left.set_ylabel(r"margin half-width $r_m$")
    left.grid(False)
    left.text(0.04, 0.95, f"{stats['refuge_mass_gt_0p1']} of {stats['total_points']} points ({stats['refuge_percent']:.1f}%)\nhave $M_{{u,\\rm int}}>0.1$", transform=left.transAxes, va="top", fontsize=8.2)
    left.text(0.05, 0.06, "A", transform=left.transAxes, fontsize=11, fontweight="bold")

    clean_axis(right)
    selected = mass.ravel() > 0.25
    A, Ymean, Sbar = access.ravel()[selected], mean.ravel()[selected], supply.ravel()[selected]
    blues = LinearSegmentedColormap.from_list("custom_blues", plt.cm.Blues(np.linspace(0.30, 1.0, 256)))
    grid = np.linspace(np.log(0.12), np.log(2.0), 300)
    right.plot(np.exp(grid), monotone_curve(grid, *parameters), color=INK, lw=1.4, label="fitted curve")
    scatter = right.scatter(A, Ymean, c=Sbar, cmap=blues, s=22, edgecolor="white", linewidth=0.4)
    colourbar = figure.colorbar(scatter, ax=right, pad=0.02)
    colourbar.set_label(r"well-mixed supply $\overline{\sigma_0}$", fontsize=9)
    right.set_xscale("log")
    right.set_xlim(0.12, 2.0)
    right.set_ylim(0, 0.85)
    right.set_xticks([0.2, 0.3, 0.5, 1.0, 2.0])
    right.set_xticklabels(["0.2", "0.3", "0.5", "1.0", "2.0"])
    right.xaxis.set_minor_formatter(NullFormatter())
    right.set_xlabel(r"immune access number $\mathcal{A}=1/(\kappa_*a)$")
    right.set_ylabel(r"interior mean tumour density $\bar{u}_{\rm int}$")
    right.legend(loc="upper right", fontsize=8)
    right.text(0.05, 0.06, "B", transform=right.transAxes, fontsize=11, fontweight="bold")
    right.text(0.05, 0.19, f"{stats['plotted_mass_gt_0p25']} out-of-fit points\nRMSE {stats['rmse']:.3f}; SD {stats['residual_std']:.3f}; mean {stats['residual_mean']:.3f}", transform=right.transAxes, fontsize=8.2, color=MUTE)
    figure.tight_layout(w_pad=2.0)
    figure.savefig(FIGURES / "fig8_refuge_phase.pdf")
    figure.savefig(FIGURES / "fig8_refuge_phase.png", dpi=170)
    plt.close(figure)


def generate_figure11():
    profiles = np.load(RESULTS / "sweep_profiles.npz")
    rows = load_json(RESULTS / "sweeps.json")
    xi_rows = sorted((row for row in rows if row["sweep"] == "xi"), key=lambda row: row["xi"])
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.4, 3.9))
    clean_axis(left)
    clean_axis(right)
    xis = [0, 4, 8, 16]
    colours = plt.cm.Blues(np.linspace(0.45, 0.95, len(xis)))
    L, rm, a = 10.0, 2.0, 3.0
    for xi, colour in zip(xis, colours):
        key = str(xi)
        x, v = profiles[f"x_{key}"], profiles[f"v_{key}"]
        kappa = float(profiles[f"kappa_{key}"])
        left.plot(x, v, color=colour, lw=1.4, label=rf"$\xi={xi}$")
        interior = (x > rm) & (x < L - rm)
        edge_index = int(np.argmin(np.abs(x - rm)))
        distance = a - np.abs(x[interior] - L / 2.0)
        barrier = v[edge_index] * np.exp(-kappa * distance)
        left.plot(x[interior], barrier, color=colour, lw=0.9, ls="--")
    left.axvspan(0, 2, color=ORANGE, alpha=0.10, lw=0)
    left.axvspan(8, 10, color=ORANGE, alpha=0.10, lw=0)
    left.set_xlabel(r"$x$")
    left.set_ylabel(r"effector density $v$")
    left.legend(fontsize=8, loc="upper center", ncol=4, columnspacing=1.0)
    left.text(0.03, 0.05, "A", transform=left.transAxes, fontsize=11, fontweight="bold")
    left.text(0.50, 0.62, r"dashed: $e^{-\kappa_*\,\mathrm{dist}}$ from the margin", transform=left.transAxes, ha="center", fontsize=8.2, color=MUTE)

    xi_values = np.array([row["xi"] for row in xi_rows])
    penetration = np.array([1.0 / row["kappa"] for row in xi_rows])
    right.plot(xi_values, penetration, "o-", color=INK, lw=1.3, ms=4)
    right.axhline(a, color=TEAL, ls=":", lw=1.0)
    right.text(24, a + 0.12, r"$a=3$: $\mathcal{A}=1$", ha="right", fontsize=8.5, color=TEAL)
    right.set_xlabel(r"chemotactic sensitivity $\xi$")
    right.set_ylabel(r"immune penetration depth $\kappa_*^{-1}$")
    right.text(0.03, 0.90, "B", transform=right.transAxes, fontsize=11, fontweight="bold")
    figure.tight_layout(w_pad=2.0)
    figure.savefig(FIGURES / "fig11_barrier.pdf")
    figure.savefig(FIGURES / "fig11_barrier.png", dpi=170)
    plt.close(figure)


def generate_all():
    apply_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    generate_figure9()
    generate_figure8()
    generate_figure11()


if __name__ == "__main__":
    generate_all()
