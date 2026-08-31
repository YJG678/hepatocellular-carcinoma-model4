"""End-to-end regeneration of every numerical value, table and figure.

Usage
-----
python reproduce.py --force
    Delete only this archive's generated results/ and figures/ directories,
    recompute every experiment, verify manuscript values, and regenerate plots.

python reproduce.py --force --skip-2d
    Recompute the one-dimensional and threshold results only.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from analysis_common import fit_monotone, write_csv, write_json
from config import (
    BASE_1D,
    BASE_2D,
    D2_GRID,
    FIGURES,
    L_GRID,
    RESULTS,
    RM_GRID,
    SEED_2D,
    SIGMA_HI_GRID,
    STATIONARITY_TOL,
    THIN_MARGIN_GRID,
    VOLUME_FILLING_VMAX,
    XI_GRID,
)
from mesh_convergence import generate as generate_mesh_convergence
from threshold_analysis import generate as generate_threshold
from tme_2d import run as run2d
from tme_core import ode_comparator, run1d, weighted_L1


def log(message):
    print(message, flush=True)


def safe_clean(directory):
    directory = Path(directory).resolve()
    root = Path(__file__).resolve().parent
    if directory.parent != root or directory.name not in {"results", "figures"}:
        raise RuntimeError(f"Refusing to clean unexpected directory: {directory}")
    if directory.exists():
        shutil.rmtree(directory)


def compact_1d(result, sweep=None):
    row = dict(
        sweep=sweep,
        xi=result["params"]["xi"],
        d2=result["params"]["d2"],
        L=result["params"]["L"],
        N=result["params"]["N"],
        T=result["params"]["T"],
        rm=result["params"]["rm"],
        sigma0_hi=result["params"]["sigma0_hi"],
        a=result["a"],
        s0bar=result["s0bar"],
        K1=result["K1"],
        K2=result["K2"],
        kappa=result["kappa"],
        A=result["A"],
        mass_int=result["mass_int"],
        mean_int=result["mean_int"],
        vmean_int=result["vmean_int"],
        mmin=result["mmin"],
        stationarity_u=result["stationarity"]["u"],
        stationarity_v=result["stationarity"]["v"],
        stationarity_w=result["stationarity"]["w"],
        stationarity_max=result["stationarity"]["max"],
        comparison_interval=result["stationarity"]["interval"],
        solver_success=result["ok"],
        nfev=result["nfev"],
        njev=result["njev"],
        nlu=result["nlu"],
    )
    return row


def generate_sweeps():
    log("[3/7] Running the 27 one-parameter transport sweeps")
    rows = []
    profiles = {}

    def add(sweep, **overrides):
        parameters = {**BASE_1D, **overrides}
        result = run1d(**parameters)
        estimate = weighted_L1(result, delta=parameters["delta"], L=parameters["L"])
        b = 1.5
        centre_mask = np.abs(result["x"] - parameters["L"] / 2.0) <= b
        row = compact_1d(result, sweep=sweep)
        row.update(
            sup_v_b=float(result["v"][centre_mask].max()),
            weighted_l1_lhs=estimate["lhs"],
            weighted_l1_rhs=estimate["rhs"],
            weighted_l1_ratio=estimate["ratio"],
        )
        rows.append(row)
        if sweep == "xi":
            profiles[str(int(parameters["xi"]))] = dict(
                x=result["x"].copy(),
                v=result["v"].copy(),
                kappa=result["kappa"],
            )

    for xi in XI_GRID:
        add("xi", xi=xi)
    for length in L_GRID:
        add(
            "a",
            L=length,
            xi=8.0,
            sigma0_hi=0.64 * length / (2.0 * BASE_1D["rm"]),
        )
    for d2 in D2_GRID:
        add("d2", d2=d2, xi=8.0)

    access = np.array([row["A"] for row in rows])
    response = np.array([row["mean_int"] for row in rows])
    fit = fit_monotone(access, response, initial_xc=0.8)
    predictors = dict(
        xi=np.array([row["xi"] for row in rows], dtype=float),
        penetration_depth=np.array([1.0 / row["kappa"] for row in rows]),
        d2=np.array([row["d2"] for row in rows]),
        a=np.array([row["a"] for row in rows]),
    )
    null_fits = {}
    for name, predictor in predictors.items():
        floor = 1.0e-9 if name == "xi" else None
        positive = np.maximum(predictor, floor) if floor is not None else predictor
        candidate = fit_monotone(
            predictor,
            response,
            zero_floor=floor,
            initial_xc=float(np.median(positive)),
        )
        null_fits[name] = dict(
            r2=candidate["r2"],
            r2_display=max(candidate["r2"], 0.0),
            y0=candidate["y0"],
            log_xc=candidate["log_xc"],
            xc=candidate["xc"],
            scale=candidate["scale"],
            zero_floor=floor,
        )

    fit_summary = dict(
        y0=fit["y0"],
        log_Ac=fit["log_xc"],
        Ac=fit["xc"],
        scale=fit["scale"],
        r2=fit["r2"],
        rmse=fit["rmse"],
        residual_mean=fit["residual_mean"],
        residual_std=fit["residual_std"],
        initial_parameters=[0.9, float(np.log(0.8)), 0.5],
        max_function_evaluations=40000,
        null_fits=null_fits,
        weighted_l1_ratio_min=min(row["weighted_l1_ratio"] for row in rows),
        weighted_l1_ratio_max=max(row["weighted_l1_ratio"] for row in rows),
        max_stationarity=max(row["stationarity_max"] for row in rows),
    )
    write_json(RESULTS / "sweeps.json", rows)
    write_csv(RESULTS / "sweeps.csv", rows)
    write_json(RESULTS / "collapse_fit.json", fit_summary)
    np.savez_compressed(
        RESULTS / "sweep_profiles.npz",
        **{
            f"x_{key}": value["x"]
            for key, value in profiles.items()
        },
        **{
            f"v_{key}": value["v"]
            for key, value in profiles.items()
        },
        **{f"kappa_{key}": value["kappa"] for key, value in profiles.items()},
    )
    return rows, fit_summary


def generate_phase_grid(fit_summary):
    log("[4/7] Running the 13 x 11 recruitment phase grid")
    rows = []
    for rm in RM_GRID:
        for sigma_hi in SIGMA_HI_GRID:
            result = run1d(
                **{
                    **BASE_1D,
                    "sigma0_hi": sigma_hi,
                    "rm": rm,
                    "T": 300.0,
                    "N": 200,
                }
            )
            row = compact_1d(result, sweep="phase")
            rows.append(row)

    mass = np.array([row["mass_int"] for row in rows])
    mean = np.array([row["mean_int"] for row in rows])
    access = np.array([row["A"] for row in rows])
    supply = np.array([row["s0bar"] for row in rows])
    keep = mass > 0.25
    parameters = [fit_summary["y0"], fit_summary["log_Ac"], fit_summary["scale"]]
    from analysis_common import monotone_curve

    predicted = monotone_curve(np.log(access[keep]), *parameters)
    residual = mean[keep] - predicted
    order = np.argsort(supply[keep])
    strata = []
    for index, selected in enumerate(np.array_split(order, 6), start=1):
        stratum_residual = residual[selected]
        stratum_supply = supply[keep][selected]
        strata.append(
            dict(
                stratum=index,
                n=len(selected),
                supply_min=float(stratum_supply.min()),
                supply_max=float(stratum_supply.max()),
                residual_mean=float(stratum_residual.mean()),
                residual_std=float(stratum_residual.std()),
                rmse=float(np.sqrt(np.mean(stratum_residual**2))),
            )
        )
    stats = dict(
        total_points=len(rows),
        refuge_mass_gt_0p1=int(np.sum(mass > 0.1)),
        refuge_percent=100.0 * float(np.mean(mass > 0.1)),
        plotted_mass_gt_0p25=int(np.sum(keep)),
        supply_range_ratio=float(supply.max() / supply.min()),
        residual_mean=float(residual.mean()),
        residual_std=float(residual.std()),
        rmse=float(np.sqrt(np.mean(residual**2))),
        max_stationarity=max(row["stationarity_max"] for row in rows),
        supply_strata=strata,
    )
    write_json(RESULTS / "phase_grid.json", rows)
    write_csv(RESULTS / "phase_grid.csv", rows)
    write_json(RESULTS / "phase_grid_stats.json", stats)
    np.savez_compressed(
        RESULTS / "phase_grid.npz",
        sigma_hi=np.array(SIGMA_HI_GRID),
        rm=np.array(RM_GRID),
        mass=mass.reshape(len(RM_GRID), len(SIGMA_HI_GRID)),
        mean=mean.reshape(len(RM_GRID), len(SIGMA_HI_GRID)),
        access=access.reshape(len(RM_GRID), len(SIGMA_HI_GRID)),
        supply=supply.reshape(len(RM_GRID), len(SIGMA_HI_GRID)),
    )
    return rows, stats


def generate_thin_margin():
    log("[2/7] Running the five-point thin-margin influx sequence")
    S = 3.2
    length = 8.0
    d2 = 0.08
    delta = 0.5
    kappa = np.sqrt(delta / d2)
    wall_value = (S / (d2 * kappa)) / np.tanh(kappa * length / 2.0)
    rows = []
    for rm, N in THIN_MARGIN_GRID:
        result = run1d(
            **{
                **BASE_1D,
                "L": length,
                "N": N,
                "T": 400.0,
                "sigma0_hi": S / rm,
                "rm": rm,
                "sigma1": 0.0,
                "xi": 0.0,
            }
        )
        maximum = float(result["v"].max())
        rows.append(
            dict(
                rm=rm,
                N=N,
                dx=result["dx"],
                sigma0_hi=S / rm,
                max_v=maximum,
                wall_value=float(wall_value),
                gap=float(wall_value - maximum),
                gap_over_rm=float((wall_value - maximum) / rm),
                stationarity=result["stationarity"]["max"],
            )
        )
    for index, row in enumerate(rows):
        row["successive_gap_ratio"] = (
            rows[index]["gap"] / rows[index + 1]["gap"]
            if index < len(rows) - 1
            else None
        )
    summary = dict(
        kappa=float(kappa),
        wall_value=float(wall_value),
        limiting_coefficient=S / (2.0 * d2),
        max_stationarity=max(row["stationarity"] for row in rows),
        successive_gap_ratios=[row["successive_gap_ratio"] for row in rows[:-1]],
    )
    write_csv(RESULTS / "thin_margin.csv", rows)
    write_json(RESULTS / "thin_margin_summary.json", summary)
    return rows, summary


def generate_2d_main():
    log("[6/7] Running the stochastic 2-D field and fixed-vmax packed comparison")
    unfilled = run2d(N=128, **BASE_2D)
    packed = run2d(N=128, **{**BASE_2D, "vmax": VOLUME_FILLING_VMAX})
    comparator = ode_comparator(unfilled["s0bar"], T=200.0)

    names = ["max_u", "max_v", "K1", "K2"]
    base_values = [
        float(unfilled["u"].max()),
        float(unfilled["v"].max()),
        unfilled["K1"],
        unfilled["K2"],
    ]
    packed_values = [
        float(packed["u"].max()),
        float(packed["v"].max()),
        packed["K1"],
        packed["K2"],
    ]
    changes = {
        name: 100.0 * (new - old) / old
        for name, old, new in zip(names, base_values, packed_values)
    }
    summary = dict(
        seed=SEED_2D,
        unfilled=dict(
            max_u=base_values[0],
            max_v=base_values[1],
            K1=base_values[2],
            K2=base_values[3],
            mmin=unfilled["mmin"],
            stationarity=unfilled["stationarity"],
            s0bar=unfilled["s0bar"],
            initial_means=unfilled["initial_means"],
            steps=unfilled["steps"],
        ),
        continuum_s0bar=1.6 * (100.0 - 36.0) / 100.0,
        ode_comparator=comparator,
        volume_filling=dict(
            q="max(v,0)*max(1-v/vmax,0)",
            vmax=VOLUME_FILLING_VMAX,
            vmax_over_unfilled_peak=VOLUME_FILLING_VMAX / base_values[1],
            max_u=packed_values[0],
            max_v=packed_values[1],
            K1=packed_values[2],
            K2=packed_values[3],
            mmin=packed["mmin"],
            stationarity=packed["stationarity"],
            percent_changes=changes,
        ),
    )
    write_json(RESULTS / "two_dimensional_summary.json", summary)
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
    return summary


def generate_summary(threshold, thin, fit, phase, mesh, two_d):
    summary = dict(
        threshold=threshold,
        thin_margin=thin,
        collapse_fit=fit,
        phase_grid=phase,
        mesh=mesh,
        two_dimensional=two_d,
        stationarity_tolerance=STATIONARITY_TOL,
    )
    write_json(RESULTS / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="remove generated outputs first")
    parser.add_argument("--skip-2d", action="store_true", help="skip all 2-D simulations")
    parser.add_argument("--no-figures", action="store_true", help="do not generate figures")
    parser.add_argument("--figures-only", action="store_true", help="use existing results")
    args = parser.parse_args()

    if args.force:
        safe_clean(RESULTS)
        safe_clean(FIGURES)
    RESULTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    if args.figures_only:
        from generate_figures import generate_all

        generate_all()
        return

    started = time.time()
    log("[1/7] Computing the analytic and mesh-converged simulated thresholds")
    _, threshold_summary = generate_threshold(RESULTS)
    _, thin_summary = generate_thin_margin()
    _, fit_summary = generate_sweeps()
    _, phase_stats = generate_phase_grid(fit_summary)
    log("[5/7] Running the four-level mesh-convergence studies")
    _, _, mesh_summary = generate_mesh_convergence(
        RESULTS, include_2d=not args.skip_2d
    )
    two_d_summary = None if args.skip_2d else generate_2d_main()
    summary = generate_summary(
        threshold_summary,
        thin_summary,
        fit_summary,
        phase_stats,
        mesh_summary,
        two_d_summary,
    )

    log("[7/7] Verifying manuscript values")
    from verify_results import verify

    verify(summary, require_2d=not args.skip_2d)
    if not args.no_figures:
        from generate_figures import generate_all

        generate_all(skip_2d=args.skip_2d)
    elapsed = time.time() - started
    (RESULTS / "runtime.txt").write_text(f"{elapsed:.6f}\n", encoding="utf-8")
    log(f"Complete in {elapsed:.1f} s. Results: {RESULTS}; figures: {FIGURES}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise

