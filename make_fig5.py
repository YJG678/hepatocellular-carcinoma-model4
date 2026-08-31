"""Standalone generation of the stochastic 128^2 field used in Fig. 5."""

import numpy as np

from config import BASE_2D, RESULTS, SEED_2D, VOLUME_FILLING_VMAX
from tme_2d import run


def generate():
    RESULTS.mkdir(parents=True, exist_ok=True)
    unfilled = run(N=128, **BASE_2D)
    packed = run(N=128, **{**BASE_2D, "vmax": VOLUME_FILLING_VMAX})
    np.savez_compressed(
        RESULTS / "fields2d.npz",
        u=unfilled["u"],
        v=unfilled["v"],
        w=unfilled["w"],
        u_packed=packed["u"],
        v_packed=packed["v"],
        w_packed=packed["w"],
        s0=unfilled["s0"],
        dx=unfilled["dx"],
        seed=SEED_2D,
        vmax=VOLUME_FILLING_VMAX,
    )
    print(f"saved {RESULTS / 'fields2d.npz'}")


if __name__ == "__main__":
    generate()

