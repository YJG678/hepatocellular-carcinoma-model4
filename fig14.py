"""Generate Fig. 14 from threshold and thin-margin result files."""

from __future__ import annotations

import csv
import json

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize_scalar

from config import FIGURES, RESULTS
from plot_style import BLUE, INK, MUTE, ORANGE, TEAL, apply_style, clean_axis


TUMOUR = "#8C2F39"
AMBER = "#8A5F10"


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_csv(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def generate():
    apply_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    threshold = load_json(RESULTS / "threshold_summary.json")
    threshold_rows = load_csv(RESULTS / "threshold_convergence.csv")
    thin_rows = load_csv(RESULTS / "thin_margin.csv")
    analytic = threshold["analytic"]
    d1, d2, delta, S = analytic["d1"], analytic["d2"], analytic["delta"], analytic["S"]
    kappa = analytic["kappa"]
    coefficient_a = analytic["coefficient_A"]
    coefficient_b = analytic["coefficient_B"]

    def G(b, a):
        return coefficient_a / b**2 + coefficient_b * np.cosh(kappa * b) / np.sinh(kappa * a)

    def min_G(a):
        result = minimize_scalar(
            lambda b: G(b, a),
            bounds=(1.0e-4, 0.999 * a),
            method="bounded",
            options=dict(xatol=1.0e-13),
        )
        return result.fun, result.x

    ac = analytic["a_proved"]
    bc = analytic["b_star"]
    derivative = analytic["g_derivative"]
    A_proved = analytic["A_proved"]
    A_sim = threshold["finest"]["A_sim"]

    figure, axes = plt.subplots(2, 2, figsize=(9.4, 6.4))
    axis = axes[0, 0]
    for a, colour, linestyle in ((1.2, AMBER, ":"), (ac, TUMOUR, "-"), (2.4, TEAL, "--")):
        bgrid = np.linspace(0.05, 0.999 * a, 600)
        label = rf"$a={a:.3f}$" if abs(a - ac) > 1e-6 else rf"$a=a_c={ac:.3f}$"
        axis.plot(bgrid, G(bgrid, a), color=colour, ls=linestyle, lw=1.6, label=label)
        minimum, minimiser = min_G(a)
        axis.plot([minimiser], [minimum], "o", color=colour, ms=5)
    axis.axhline(1.0, color=MUTE, lw=0.9, ls="-.")
    axis.axvline(bc, color=MUTE, lw=0.8, ls=":")
    axis.set_xlim(0, 2.4)
    axis.set_ylim(0, 3.2)
    axis.set_xlabel(r"refuge half-width $b$")
    axis.set_ylabel(r"$G(b;a)$")
    axis.set_title(r"(A) strictly convex in $b$: unique minimiser", loc="left", fontsize=9.5)
    axis.legend(fontsize=8, loc="upper right")

    axis = axes[0, 1]
    agrid = np.linspace(0.55, 4.0, 500)
    gvalues = np.array([min_G(a)[0] for a in agrid])
    axis.plot(agrid, gvalues, color=BLUE, lw=1.8)
    axis.axhline(1.0, color=MUTE, lw=0.9, ls="-.")
    axis.plot([ac], [1.0], "o", color=TUMOUR, ms=6)
    tangent = np.linspace(ac - 0.5, ac + 0.5, 2)
    axis.plot(tangent, 1.0 + derivative * (tangent - ac), color=TUMOUR, lw=1.0, ls="--")
    axis.annotate(
        rf"$a_c={ac:.4f}$" + "\n" + rf"$g'(a_c)={derivative:.3f}$",
        xy=(ac, 1.0),
        xytext=(ac + 0.52, 1.75),
        fontsize=8.2,
        color=TUMOUR,
        arrowprops=dict(arrowstyle="-", color=TUMOUR, lw=0.7),
    )
    axis.fill_between(agrid, 0, 1, where=gvalues < 1, color=TUMOUR, alpha=0.10)
    axis.set_xlim(0.55, 4.0)
    axis.set_ylim(0, 3.2)
    axis.set_xlabel(r"half-domain $a=L/2$")
    axis.set_ylabel(r"$g(a)=\min_bG(b;a)$")
    axis.set_title(r"(B) $g$ strictly decreasing: unique root", loc="left", fontsize=9.5)

    axis = axes[1, 0]
    access_grid = 1.0 / (kappa * agrid)
    order = np.argsort(access_grid)
    axis.plot(access_grid[order], gvalues[order], color=BLUE, lw=1.8)
    axis.axhline(1.0, color=MUTE, lw=0.9, ls="-.")
    axis.axvline(A_proved, color=TUMOUR, lw=1.2, ls="--")
    axis.axvline(A_sim, color=TEAL, lw=1.2, ls=":")
    axis.text(A_proved - 0.008, 3.95, rf"$\mathcal{{A}}_c^{{\rm proved}}={A_proved:.3f}$", color=TUMOUR, fontsize=8.5, rotation=90, va="top", ha="right")
    axis.text(A_sim + 0.010, 3.95, rf"$\mathcal{{A}}_c^{{\rm sim}}={A_sim:.3f}$", color=TEAL, fontsize=8.5, rotation=90, va="top")
    axis.annotate("", xy=(A_proved, 0.50), xytext=(A_sim, 0.50), arrowprops=dict(arrowstyle="<->", color=MUTE, lw=0.9))
    axis.text(0.30, 0.42, f"conservative\nby {threshold['discrepancy_percent']:.1f}%", fontsize=8, color=MUTE)
    axis.set_xlim(0.10, 0.50)
    axis.set_ylim(0, 4.0)
    axis.set_xlabel(r"immune access number $\mathcal{A}=1/(\kappa a)$")
    axis.set_ylabel(r"$g$")
    axis.set_title("(C) proved and simulated linear-onset thresholds", loc="left", fontsize=9.5)

    axis = axes[1, 1]
    rm = np.array([float(row["rm"]) for row in thin_rows])
    vmax = np.array([float(row["max_v"]) for row in thin_rows])
    gap = np.array([float(row["gap"]) for row in thin_rows])
    wall = float(thin_rows[0]["wall_value"])
    axis.semilogx(rm, vmax, "o-", color=BLUE, lw=1.5, ms=5, label=r"PDE: $\max_xv$")
    axis.axhline(wall, color=TUMOUR, lw=1.3, ls="--", label=rf"exact wall value $={wall:.2f}$")
    axis.set_xlabel(r"margin width $r_m$ at fixed $S=3.2$")
    axis.set_ylabel("peak effector density")
    axis.set_ylim(4, 17.5)
    axis.invert_xaxis()
    axis.legend(fontsize=8, loc="lower right")
    axis.set_title("(D) thin-margin sequence", loc="left", fontsize=9.5)
    inset = axis.inset_axes([0.16, 0.58, 0.32, 0.33])
    inset.loglog(rm, gap, "s-", color=TEAL, ms=3.5, lw=1.1)
    inset.loglog(rm, (S / (2.0 * d2)) * rm, color=MUTE, lw=0.8, ls=":")
    inset.text(0.55, 1.3, r"$20r_m$", fontsize=7, color=MUTE)
    inset.tick_params(labelsize=6)
    inset.set_xlabel(r"$r_m$", fontsize=6, labelpad=0)
    inset.set_ylabel("gap", fontsize=6, labelpad=1)
    inset.invert_xaxis()

    for item in axes.ravel():
        clean_axis(item)
    figure.tight_layout()
    figure.savefig(FIGURES / "fig14_sharp_threshold.pdf")
    figure.savefig(FIGURES / "fig14_sharp_threshold.png", dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    generate()
