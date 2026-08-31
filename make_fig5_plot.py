"""Plot Fig. 5 from results/fields2d.npz; performs no simulation."""

import numpy as np
import matplotlib.pyplot as plt

from config import FIGURES, RESULTS
from plot_style import MUTE, apply_style


def generate():
    apply_style()
    FIGURES.mkdir(parents=True, exist_ok=True)
    fields = np.load(RESULTS / "fields2d.npz")
    u, v, w = fields["u"], fields["v"], fields["w"]
    figure, axes = plt.subplots(1, 3, figsize=(10.0, 3.3), squeeze=False)
    for column, (field, label, colourmap) in enumerate(
        [
            (u, r"tumour $u$", "OrRd"),
            (np.log10(np.maximum(v, 1.0e-6)), r"effector $\log_{10}v$", "Blues"),
            (w, r"chemokine $w$", "BuPu"),
        ]
    ):
        axis = axes[0, column]
        image = axis.imshow(field.T, origin="lower", extent=[0, 10, 0, 10], cmap=colourmap, rasterized=True)
        colourbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.03)
        colourbar.outline.set_linewidth(0.6)
        colourbar.ax.tick_params(labelsize=7.5, colors=MUTE)
        axis.set_title(label, fontsize=9.5, pad=5)
        axis.set_xticks([0, 5, 10])
        axis.set_yticks([0, 5, 10])
        axis.tick_params(labelsize=8)
    axes[0, 0].set_ylabel(r"margin-limited recruitment, $\xi=8$", fontsize=9.5)
    figure.tight_layout(w_pad=1.4)
    figure.savefig(FIGURES / "fig5_2d_phenotypes.pdf")
    figure.savefig(FIGURES / "fig5_2d_phenotypes.png", dpi=170)
    plt.close(figure)


if __name__ == "__main__":
    generate()

