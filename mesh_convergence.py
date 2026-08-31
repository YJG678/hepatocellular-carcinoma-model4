"""Four-level one- and two-dimensional convergence studies."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from analysis_common import write_csv, write_json
from config import (
    BASE_1D,
    BASE_2D,
    MESH_1D_GRID,
    MESH_2D_GRID,
    RESULTS,
    SEED_2D,
)
from tme_2d import run as run2d
from tme_core import run1d


def relative_error(coarse, fine_restricted):
    return float(
        np.linalg.norm(coarse - fine_restricted)
        / max(np.linalg.norm(fine_restricted), 1.0e-30)
    )


def run_1d(output_dir=RESULTS):
    solutions = {}
    rows = []
    for N in MESH_1D_GRID:
        result = run1d(**{**BASE_1D, "N": N})
        solutions[N] = result
        rows.append(
            dict(
                N=N,
                dx=result["dx"],
                mean_u_int=result["mean_int"],
                mean_v_int=result["vmean_int"],
                access_number=result["A"],
                E_u=None,
                E_v=None,
                E_w=None,
                stationarity=result["stationarity"]["max"],
                s0bar=result["s0bar"],
            )
        )
    for row, N in zip(rows[:-1], MESH_1D_GRID[:-1]):
        coarse, fine = solutions[N], solutions[2 * N]
        for field, key in (("u", "E_u"), ("v", "E_v"), ("w", "E_w")):
            restricted = fine[field].reshape(N, 2).mean(axis=1)
            row[key] = relative_error(coarse[field], restricted)
    write_csv(output_dir / "mesh_convergence_1d.csv", rows)
    return rows


def run_2d(output_dir=RESULTS):
    solutions = {}
    rows = []
    mesh_parameters = {
        **BASE_2D,
        "noise_amp": 0.0,
        "zero_mean_noise": False,
        "seed": SEED_2D,
    }
    for N in MESH_2D_GRID:
        result = run2d(N=N, **mesh_parameters)
        solutions[N] = result
        rows.append(
            dict(
                N=N,
                dx=result["dx"],
                max_u=float(result["u"].max()),
                max_v=float(result["v"].max()),
                E_u=None,
                E_v=None,
                E_w=None,
                stationarity=result["stationarity"]["max"],
                s0bar=result["s0bar"],
                K1=result["K1"],
                K2=result["K2"],
            )
        )
    for row, N in zip(rows[:-1], MESH_2D_GRID[:-1]):
        coarse, fine = solutions[N], solutions[2 * N]
        for field, key in (("u", "E_u"), ("v", "E_v"), ("w", "E_w")):
            restricted = fine[field].reshape(N, 2, N, 2).mean(axis=(1, 3))
            row[key] = relative_error(coarse[field], restricted)
    write_csv(output_dir / "mesh_convergence_2d.csv", rows)
    return rows


def empirical_orders(rows, key):
    errors = [row[key] for row in rows if row[key] is not None]
    return [float(np.log2(errors[i] / errors[i + 1])) for i in range(len(errors) - 1)]


def generate(output_dir=RESULTS, include_2d=True):
    rows_1d = run_1d(output_dir)
    rows_2d = run_2d(output_dir) if include_2d else []
    summary = dict(
        orders_1d={field: empirical_orders(rows_1d, field) for field in ("E_u", "E_v", "E_w")},
        orders_2d={field: empirical_orders(rows_2d, field) for field in ("E_u", "E_v", "E_w")}
        if rows_2d
        else {},
    )
    if rows_2d:
        coarse, fine = rows_2d[-2], rows_2d[-1]
        summary["finest_peak_changes_percent"] = dict(
            u=100.0 * abs(fine["max_u"] - coarse["max_u"]) / abs(fine["max_u"]),
            v=100.0 * abs(fine["max_v"] - coarse["max_v"]) / abs(fine["max_v"]),
        )
    write_json(output_dir / "mesh_convergence_summary.json", summary)
    return rows_1d, rows_2d, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(RESULTS))
    parser.add_argument("--skip-2d", action="store_true")
    args = parser.parse_args()
    generate(Path(args.output), include_2d=not args.skip_2d)


if __name__ == "__main__":
    main()

