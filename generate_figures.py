"""Regenerate every manuscript figure from machine-readable results."""

from config import FIGURES
from fig14 import generate as generate_figure14
from figures_8_9_11 import generate_all as generate_8_9_11
from make_fig5_plot import generate as generate_figure5


def generate_all(skip_2d=False):
    FIGURES.mkdir(parents=True, exist_ok=True)
    generate_8_9_11()
    generate_figure14()
    if not skip_2d:
        generate_figure5()
    expected = [
        "fig8_refuge_phase.pdf",
        "fig8_refuge_phase.png",
        "fig9_access_number.pdf",
        "fig9_access_number.png",
        "fig11_barrier.pdf",
        "fig11_barrier.png",
        "fig14_sharp_threshold.pdf",
        "fig14_sharp_threshold.png",
    ]
    if not skip_2d:
        expected.extend(["fig5_2d_phenotypes.pdf", "fig5_2d_phenotypes.png"])
    for name in expected:
        path = FIGURES / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"Figure generation failed or produced an empty file: {path}")
        if path.suffix == ".pdf" and not path.read_bytes().startswith(b"%PDF"):
            raise RuntimeError(f"Invalid PDF header: {path}")
    print(f"figures written to {FIGURES}")


if __name__ == "__main__":
    generate_all()
