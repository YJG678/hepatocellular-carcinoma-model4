"""Mesh-converged linear-onset threshold used in Example 1."""

from __future__ import annotations

import argparse

import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.sparse import diags, eye
from scipy.sparse.linalg import eigsh, spsolve

from analysis_common import write_csv, write_json
from config import RESULTS, THRESHOLD_GRID
from tme_core import cell_average_margin_profile


D1 = 0.01
D2 = 0.08
DELTA = 0.5
S = 3.2
RM = 0.1
SIGMA_HI = S / RM
KAPPA = np.sqrt(DELTA / D2)
EIG_TOL = 2.0e-10
BISECTION_BRACKET = (1.44, 1.49)
BISECTION_WIDTH = 1.2e-8


def neumann_laplacian(N, length):
    dx = length / N
    diagonal = -2.0 * np.ones(N)
    diagonal[[0, -1]] = -1.0
    return diags(
        [np.ones(N - 1), diagonal, np.ones(N - 1)],
        [-1, 0, 1],
        format="csr",
    ) / dx**2


def largest_growth_rate(a, N):
    length = 2.0 * a
    laplacian = neumann_laplacian(N, length)
    source = cell_average_margin_profile(length, N, SIGMA_HI, RM)
    v0 = spsolve(DELTA * eye(N, format="csr") - D2 * laplacian, source)
    linearised = D1 * laplacian + diags(1.0 - v0, 0, format="csr")
    initial_vector = np.ones(N) / np.sqrt(N)
    eigenvalue = eigsh(
        linearised,
        k=1,
        which="LA",
        return_eigenvectors=False,
        tol=EIG_TOL,
        maxiter=100000,
        v0=initial_vector,
    )[0]
    return float(eigenvalue)


def bisect_threshold(N):
    lower, upper = BISECTION_BRACKET
    lower_value = largest_growth_rate(lower, N)
    upper_value = largest_growth_rate(upper, N)
    if not (lower_value < 0.0 < upper_value):
        raise RuntimeError(
            f"Threshold is not bracketed for N={N}: {lower_value}, {upper_value}"
        )
    iterations = 0
    while upper - lower >= BISECTION_WIDTH:
        midpoint = 0.5 * (lower + upper)
        value = largest_growth_rate(midpoint, N)
        if value > 0.0:
            upper = midpoint
        else:
            lower = midpoint
        iterations += 1
    threshold = 0.5 * (lower + upper)
    return dict(
        N=N,
        lower=lower,
        upper=upper,
        bracket_width=upper - lower,
        iterations=iterations,
        a_sim=threshold,
        A_sim=1.0 / (KAPPA * threshold),
        lambda_mid=largest_growth_rate(threshold, N),
    )


def analytic_threshold():
    coefficient_a = D1 * np.pi**2 / 4.0
    coefficient_b = S / (D2 * KAPPA)

    def functional(b, a):
        return coefficient_a / b**2 + coefficient_b * np.cosh(KAPPA * b) / np.sinh(
            KAPPA * a
        )

    def minimise(a):
        result = minimize_scalar(
            lambda b: functional(b, a),
            bounds=(1.0e-5, 80.0),
            method="bounded",
            options=dict(xatol=1.0e-14),
        )
        return result.fun, result.x

    a_proved = brentq(lambda a: minimise(a)[0] - 1.0, 0.2, 8.0, xtol=1.0e-13)
    g_value, b_star = minimise(a_proved)
    derivative = (
        -coefficient_b
        * KAPPA
        * np.cosh(KAPPA * b_star)
        * np.cosh(KAPPA * a_proved)
        / np.sinh(KAPPA * a_proved) ** 2
    )
    return dict(
        d1=D1,
        d2=D2,
        delta=DELTA,
        S=S,
        kappa=float(KAPPA),
        coefficient_A=float(coefficient_a),
        coefficient_B=float(coefficient_b),
        a_proved=float(a_proved),
        b_star=float(b_star),
        g_at_threshold=float(g_value),
        g_derivative=float(derivative),
        A_proved=float(1.0 / (KAPPA * a_proved)),
    )


def generate(output_dir=RESULTS):
    rows = [bisect_threshold(N) for N in THRESHOLD_GRID]
    analytic = analytic_threshold()
    finest = rows[-1]
    discrepancy = 100.0 * (finest["A_sim"] - analytic["A_proved"]) / finest["A_sim"]
    summary = dict(
        analytic=analytic,
        finest=finest,
        discrepancy_percent=float(discrepancy),
        eig_tolerance=EIG_TOL,
        bisection_bracket=BISECTION_BRACKET,
        stopping_width=BISECTION_WIDTH,
    )
    write_csv(output_dir / "threshold_convergence.csv", rows)
    write_json(output_dir / "threshold_summary.json", summary)
    return rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(RESULTS))
    args = parser.parse_args()
    rows, summary = generate(output_dir=np_path(args.output))
    for row in rows:
        print(
            f"N={row['N']:4d} a_sim={row['a_sim']:.8f} "
            f"A_sim={row['A_sim']:.8f} width={row['bracket_width']:.3e}"
        )
    print(f"discrepancy = {summary['discrepancy_percent']:.6f}%")


def np_path(value):
    from pathlib import Path

    return Path(value)


if __name__ == "__main__":
    main()

