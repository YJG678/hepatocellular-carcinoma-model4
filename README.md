# HCC TME numerical reproducibility archive

This archive regenerates every numerical value, table and figure in Example 1
and the revised `Numerical results` section of the manuscript.

## Clean reproduction

Create an isolated Python environment, install the pinned dependencies, and run:

```bash
python -m pip install -r requirements.txt
python reproduce.py --force
```

`--force` deletes only this archive's generated `results/` and `figures/`
directories before recomputing them.  It never deletes source files.  A full
run includes the `320^2` mesh and normally takes a few minutes on a modern
workstation.  For a one-dimensional diagnostic run:

```bash
python reproduce.py --force --skip-2d
```

After a completed run, repeat the numerical assertions without simulating:

```bash
python verify_results.py
```

Regenerate plots without simulating:

```bash
python reproduce.py --figures-only
```

## Reproducibility decisions

- Every one-dimensional time-dependent run starts from `u=0.5`, `v=0.5`,
  `w=0` and is deterministic.
- The displayed stochastic two-dimensional field uses NumPy
  `default_rng(seed=1)` and perturbation amplitude `1e-3`.
- The two-dimensional mesh study suppresses random perturbations and uses
  `N=40,80,160,320`; these grids put `r_m=2` on a cell face, keeping the
  discrete source mean exactly `1.024`.
- The volume-filling calculation sets the literal parameter
  `vmax=30.41011648` and uses
  `q(v)=max(v,0)*max(1-v/vmax,0)`.
- The simulated threshold is not obtained from a finite-time tumour cutoff.
  `threshold_analysis.py` solves the discrete tumour-free effector equation,
  computes the principal growth eigenvalue, and bisects its zero.  The thin
  discontinuous source is integrated exactly over every finite volume.
- RMSE is computed as `sqrt(mean(residual**2))`; residual standard deviation is
  reported separately.
- All BDF integrations are checked for solver success and finite values.

## Source files

| File | Purpose |
| --- | --- |
| `config.py` | exact parameters, grids, seed and paths |
| `tme_core.py` | 1-D finite-volume solver, diagnostics and ODE comparator |
| `tme_2d.py` | 2-D IMEX/DCT solver and finite-packing mobility |
| `threshold_analysis.py` | Example 1 threshold and convergence table |
| `mesh_convergence.py` | four-level 1-D and 2-D mesh studies |
| `reproduce.py` | clean end-to-end simulation entry point |
| `verify_results.py` | numerical assertions against the manuscript |
| `figures_8_9_11.py` | Figs. 8, 9 and 11 from generated data |
| `fig14.py` | Fig. 14 from generated threshold/influx data |
| `make_fig5.py` | standalone generation of the Fig. 5 field data |
| `make_fig5_plot.py` | Fig. 5 plotting only |
| `generate_figures.py` | regenerate all figures from saved results |
| `seeds.py` | complete random-seed registry |

No plotting script contains manuscript result arrays.  Analytical model
parameters and plotting coordinates may be specified in plotting code, but all
reported simulation outputs are read from `results/`.

`manuscript_sync_patch.tex` records four small wording/rounding corrections
identified while validating the newly revised manuscript excerpt against the
executable definitions.

## Generated results

The clean run creates, among others:

- `threshold_convergence.csv`, `threshold_summary.json`;
- `thin_margin.csv`;
- `sweeps.csv`, `sweeps.json`, `collapse_fit.json`;
- `phase_grid.csv`, `phase_grid.npz`, `phase_grid_stats.json`;
- `mesh_convergence_1d.csv`, `mesh_convergence_2d.csv`;
- `fields2d.npz`, `two_dimensional_summary.json`;
- `summary.json` and `verification_report.txt`;
- publication PNG and PDF figures under `figures/`.

## Expected checks

`verify_results.py` checks the unrounded values behind the manuscript's
reported values, including:

- `A_c^proved=0.24811689`, `a_c^sim=1.46300268`,
  `A_c^sim=0.27341030`, discrepancy `9.251%`;
- collapse `R^2=0.987679`, RMSE `0.017161`;
- phase-grid counts `135/143` and `122`, residual SD `0.09775`, RMSE `0.10224`;
- all two-dimensional unfilled and packed maxima and derivative diagnostics;
- stationarity, bisection iterations and grid-convergence peak changes.

Stationarity residuals are acceptance diagnostics: the verifier requires each
one to be finite, non-negative, and no larger than the declared threshold
`1e-6`.  Their last digits are intentionally not compared with a stored Linux
bit pattern because FFT, SciPy/BLAS, CPU, and operating-system builds can
produce harmless round-off differences (typically around `1e-15`).

## Licence

The source code is released under the MIT License in `LICENSE`.  Cite the
versioned release identified by `CITATION.cff` and the repository release tag
used for the submitted manuscript.
