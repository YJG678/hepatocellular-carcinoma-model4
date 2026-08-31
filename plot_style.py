"""Shared, deterministic Matplotlib style."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BLUE = "#2F6FD0"
ORANGE = "#D2691E"
TEAL = "#1E9E8A"
INK = "#1a1a1a"
MUTE = "#5a5a5a"
GRID = "#d8d8d8"


def apply_style():
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["DejaVu Serif"],
            "font.size": 9,
            "axes.edgecolor": MUTE,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": MUTE,
            "ytick.color": MUTE,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.8,
            "axes.axisbelow": True,
            "figure.dpi": 150,
        }
    )


def clean_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)

