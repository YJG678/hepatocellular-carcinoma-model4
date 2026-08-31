"""Fail loudly if regenerated outputs do not reproduce manuscript values."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from analysis_common import read_json
from config import RESULTS, STATIONARITY_TOL


def close(label, actual, expected, tolerance):
    error = abs(float(actual) - float(expected))
    if error > tolerance:
        raise AssertionError(
            f"{label}: computed {actual!r}, expected {expected!r}, "
            f"absolute error {error:.3g} > {tolerance:.3g}"
        )
    return f"PASS {label}: {float(actual):.10g}"


def exact(label, actual, expected):
    if actual != expected:
        raise AssertionError(f"{label}: computed {actual!r}, expected {expected!r}")
    return f"PASS {label}: {actual}"


def at_most(label, actual, upper_bound):
    """Check a stopping/acceptance criterion rather than a reference bit pattern.

    Residuals at the end of an iterative time integration can differ in their
    last few digits across SciPy, BLAS, FFT, CPU, and operating-system builds.
    Reproducibility therefore means satisfying the declared stationarity
    criterion, not reproducing a Linux residual bit for bit.
    """
    value = float(actual)
    if not math.isfinite(value) or value < 0.0 or value > upper_bound:
        raise AssertionError(
            f"{label}: computed {actual!r} is not a finite, non-negative "
            f"value at most {upper_bound:.3g}"
        )
    return f"PASS {label}: {value:.10g} <= {upper_bound:.3g}"


def verify(summary, require_2d=True):
    messages = []
    threshold = summary["threshold"]
    messages += [
        close("proved access threshold", threshold["analytic"]["A_proved"], 0.24811689, 5e-9),
        close("simulated half-width", threshold["finest"]["a_sim"], 1.46300268, 8e-9),
        close("simulated access threshold", threshold["finest"]["A_sim"], 0.27341030, 8e-9),
        close("threshold discrepancy percent", threshold["discrepancy_percent"], 9.251, 0.001),
        exact("threshold bisection iterations", threshold["finest"]["iterations"], 22),
    ]

    thin = summary["thin_margin"]
    messages.append(close("thin-margin wall value", thin["wall_value"], 16.0000001, 6e-8))
    expected_ratios = [1.47, 1.67, 1.80, 1.87]
    for index, (actual, expected) in enumerate(
        zip(thin["successive_gap_ratios"], expected_ratios), start=1
    ):
        messages.append(close(f"thin-margin gap ratio {index}", actual, expected, 0.006))

    fit = summary["collapse_fit"]
    messages += [
        close("fit y0", fit["y0"], 0.75373681, 8e-8),
        close("fit log Ac", fit["log_Ac"], -0.33860996, 8e-8),
        close("fit scale", fit["scale"], 0.49146793, 8e-8),
        close("fit Ac", fit["Ac"], 0.71276040, 8e-8),
        close("fit R2", fit["r2"], 0.987679, 8e-7),
        close("fit RMSE", fit["rmse"], 0.017161, 8e-7),
        close("weighted L1 ratio min", fit["weighted_l1_ratio_min"], 5.2218, 8e-5),
        close("weighted L1 ratio max", fit["weighted_l1_ratio_max"], 16.2045, 8e-5),
    ]
    expected_null = dict(
        xi=0.766627,
        penetration_depth=0.697175,
        d2=0.031612,
        a=-2.72e-7,
    )
    for name, expected in expected_null.items():
        messages.append(close(f"null-fit R2 ({name})", fit["null_fits"][name]["r2"], expected, 8e-7))

    phase = summary["phase_grid"]
    messages += [
        exact("phase-grid point count", phase["total_points"], 143),
        exact("phase-grid refuge count", phase["refuge_mass_gt_0p1"], 135),
        exact("phase-grid plotted count", phase["plotted_mass_gt_0p25"], 122),
        close("phase-grid refuge percent", phase["refuge_percent"], 94.4, 0.06),
        close("phase-grid residual mean", phase["residual_mean"], 0.02995, 8e-5),
        close("phase-grid residual SD", phase["residual_std"], 0.09775, 8e-5),
        close("phase-grid RMSE", phase["rmse"], 0.10224, 8e-5),
    ]

    mesh = summary["mesh"]
    if require_2d:
        messages += [
            close(
                "2-D finest peak-u change percent",
                mesh["finest_peak_changes_percent"]["u"],
                0.681,
                0.001,
            ),
            close(
                "2-D finest peak-v change percent",
                mesh["finest_peak_changes_percent"]["v"],
                0.051,
                0.001,
            ),
        ]
        two_d = summary["two_dimensional"]
        unfilled = two_d["unfilled"]
        packed = two_d["volume_filling"]
        messages += [
            exact("2-D random seed", two_d["seed"], 1),
            close("2-D discrete source mean", unfilled["s0bar"], 1.0359375, 1e-12),
            close("2-D max u", unfilled["max_u"], 0.37248377, 8e-9),
            close("2-D max v", unfilled["max_v"], 3.04101165, 8e-9),
            close("2-D K1", unfilled["K1"], 0.29202086, 8e-9),
            close("2-D K2", unfilled["K2"], 3.87014054, 8e-8),
            at_most("2-D stationarity", unfilled["stationarity"]["max"], STATIONARITY_TOL),
            close("volume-filling vmax", packed["vmax"], 30.41011648, 1e-12),
            close("packed max u", packed["max_u"], 0.39086441, 8e-9),
            close("packed max v", packed["max_v"], 3.04787276, 8e-9),
            close("packed K1", packed["K1"], 0.23007860, 8e-9),
            close("packed K2", packed["K2"], 2.15484132, 8e-8),
            close("packed max-u percent change", packed["percent_changes"]["max_u"], 4.935, 0.001),
            close("packed max-v percent change", packed["percent_changes"]["max_v"], 0.226, 0.001),
            close("packed K1 percent change", packed["percent_changes"]["K1"], -21.212, 0.001),
            close("packed K2 percent change", packed["percent_changes"]["K2"], -44.321, 0.001),
        ]

        with (RESULTS / "mesh_convergence_2d.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            mesh_rows = list(csv.DictReader(handle))
        expected_mesh = {
            40: (0.366880, 3.038366, 0.029955, 0.006662, 0.025812),
            80: (0.377085, 3.033917, 0.016615, 0.003278, 0.014321),
            160: (0.382267, 3.031132, 0.008757, 0.001741, 0.007545),
            320: (0.384887, 3.029583, None, None, None),
        }
        for row in mesh_rows:
            N = int(row["N"])
            max_u, max_v, eu, ev, ew = expected_mesh[N]
            messages.append(close(f"mesh N={N} max u", row["max_u"], max_u, 6e-7))
            messages.append(close(f"mesh N={N} max v", row["max_v"], max_v, 6e-7))
            for key, expected in (("E_u", eu), ("E_v", ev), ("E_w", ew)):
                if expected is not None:
                    messages.append(close(f"mesh N={N} {key}", row[key], expected, 6e-7))
            messages.append(
                at_most(
                    f"mesh N={N} stationarity",
                    row["stationarity"],
                    STATIONARITY_TOL,
                )
            )

    report = "\n".join(messages) + f"\n\nALL {len(messages)} CHECKS PASSED\n"
    (RESULTS / "verification_report.txt").write_text(report, encoding="utf-8")
    print(report, end="")
    return messages


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", default=str(RESULTS / "summary.json"))
    parser.add_argument("--skip-2d", action="store_true")
    args = parser.parse_args()
    verify(read_json(Path(args.summary)), require_2d=not args.skip_2d)


if __name__ == "__main__":
    main()
