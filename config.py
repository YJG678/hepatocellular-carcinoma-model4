"""Single source of truth for every parameter used in the manuscript."""

from pathlib import Path


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"

SEED_2D = 1
VOLUME_FILLING_VMAX = 30.41011648

BASE_1D = dict(
    L=10.0,
    N=200,
    T=400.0,
    d1=0.01,
    d2=0.08,
    d3=1.0,
    xi=8.0,
    sigma0=None,
    sigma0_hi=1.6,
    rm=2.0,
    sigma1=0.5,
    alpha=0.5,
    gamma=0.5,
    beta=1.0,
    delta=0.5,
    ell=1.0,
    vmax=None,
    u0=0.5,
    v0=0.5,
    w0=0.0,
    seed=0,
    noise=0.0,
    nsamp=60,
    rtol=1.0e-8,
    atol=1.0e-10,
)

BASE_2D = dict(
    L=10.0,
    T=200.0,
    seed=SEED_2D,
    noise_amp=1.0e-3,
    vmax=None,
    d1=0.01,
    d2=0.08,
    d3=1.0,
    xi=8.0,
    sigma0=0.0,
    sigma1=0.5,
    alpha=0.5,
    gamma=0.5,
    beta=1.0,
    delta=0.5,
    ell=1.0,
    margin=(1.6, 2.0),
    ic="margin",
    cfl=0.4,
)

XI_GRID = [0, 1, 2, 3, 4, 6, 8, 10, 12, 14, 16, 20, 24]
L_GRID = [8.0, 9.0, 10.0, 12.0, 14.0, 16.0, 20.0]
D2_GRID = [0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]
SIGMA_HI_GRID = [0.8 + 0.2 * i for i in range(13)]
RM_GRID = [0.50 + 0.35 * i for i in range(11)]

THIN_MARGIN_GRID = [
    (1.0, 3200),
    (0.5, 3200),
    (0.25, 3200),
    (0.125, 6400),
    (0.0625, 6400),
]

MESH_1D_GRID = [100, 200, 400, 800]
MESH_2D_GRID = [40, 80, 160, 320]
THRESHOLD_GRID = [400, 800, 1600, 3200]

STATIONARITY_TOL = 1.0e-6

