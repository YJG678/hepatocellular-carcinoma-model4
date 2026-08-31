"""Shared fitting, metrics and serialisation helpers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import curve_fit


def monotone_curve(log_x, y0, log_xc, scale):
    return y0 / (1.0 + np.exp((log_x - log_xc) / scale))


def fit_monotone(x, y, zero_floor=None, initial_xc=0.8):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if zero_floor is not None:
        x = np.maximum(x, zero_floor)
    if np.any(x <= 0.0):
        raise ValueError("The monotone fit requires positive predictor values")
    if initial_xc is None:
        initial_xc = float(np.median(x))
    initial = [0.9, np.log(initial_xc), 0.5]
    parameters, covariance = curve_fit(
        monotone_curve,
        np.log(x),
        y,
        p0=initial,
        maxfev=40000,
    )
    fitted = monotone_curve(np.log(x), *parameters)
    residual = y - fitted
    denominator = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(residual**2)) / denominator
    log_xc = float(parameters[1])
    xc = float(np.exp(log_xc)) if log_xc <= np.log(np.finfo(float).max) else None
    return dict(
        parameters=parameters,
        covariance=covariance,
        fitted=fitted,
        residual=residual,
        y0=float(parameters[0]),
        log_xc=log_xc,
        xc=xc,
        scale=float(parameters[2]),
        r2=float(r2),
        rmse=float(np.sqrt(np.mean(residual**2))),
        residual_mean=float(residual.mean()),
        residual_std=float(residual.std()),
    )


def jsonable(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(jsonable(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [jsonable(row) for row in rows]
    if not rows:
        raise ValueError(f"Cannot write an empty CSV file: {path}")
    if fieldnames is None:
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)
