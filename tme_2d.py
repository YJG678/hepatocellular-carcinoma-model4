"""Two-dimensional IMEX/DCT solver used by the manuscript."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.fft import dctn, idctn


def dct_diffusion_factors(N, L, diffusion, dt):
    dx = L / N
    lam = (2.0 / dx**2) * (1.0 - np.cos(np.pi * np.arange(N) / N))
    return 1.0 / (1.0 + dt * diffusion * (lam[:, None] + lam[None, :]))


def implicit_diffuse(field, factor):
    return idctn(
        dctn(field, type=2, norm="ortho") * factor,
        type=2,
        norm="ortho",
    )


def mobility(v, vmax=None):
    """q(v)=v or q(v)=v_+(1-v/vmax)_+."""
    if vmax is None:
        return v
    return np.clip(v, 0.0, None) * np.clip(1.0 - v / vmax, 0.0, None)


def chemotaxis_div(v, w, xi, dx, vmax=None):
    """Conservative upwind divergence with zero boundary flux."""
    q = mobility(v, vmax)
    dwx = (w[1:, :] - w[:-1, :]) / dx
    qx = np.where(dwx >= 0.0, q[:-1, :], q[1:, :])
    flux_x = xi * qx * dwx
    dwy = (w[:, 1:] - w[:, :-1]) / dx
    qy = np.where(dwy >= 0.0, q[:, :-1], q[:, 1:])
    flux_y = xi * qy * dwy
    div = np.zeros_like(v)
    div[:-1, :] += flux_x / dx
    div[1:, :] -= flux_x / dx
    div[:, :-1] += flux_y / dx
    div[:, 1:] -= flux_y / dx
    return div


def grad_mag_max(w, dx):
    gx = np.zeros_like(w)
    gy = np.zeros_like(w)
    gx[1:-1, :] = (w[2:, :] - w[:-2, :]) / (2.0 * dx)
    gy[:, 1:-1] = (w[:, 2:] - w[:, :-2]) / (2.0 * dx)
    return float(np.max(np.sqrt(gx**2 + gy**2)))


def lap_max(w, dx):
    padded = np.pad(w, 1, mode="edge")
    lap = (
        padded[2:, 1:-1]
        + padded[:-2, 1:-1]
        + padded[1:-1, 2:]
        + padded[1:-1, :-2]
        - 4.0 * w
    ) / dx**2
    return float(np.max(np.abs(lap)))


def relative_change(new, old):
    return float(np.linalg.norm(new - old) / max(np.linalg.norm(new), 1.0e-30))


def coexistence(sigma0, sigma1, alpha, gamma, beta, delta, ell):
    from scipy.optimize import brentq

    def residual(u):
        w = u * (alpha + gamma * (1.0 - u)) / ell
        return sigma0 + sigma1 * w / (1.0 + w) - (delta + beta * u) * (
            1.0 - u
        )

    grid = np.linspace(1.0e-9, 1.0 - 1.0e-9, 20001)
    values = np.array([residual(value) for value in grid])
    for i in range(len(grid) - 1):
        if values[i] * values[i + 1] < 0.0:
            return brentq(residual, grid[i], grid[i + 1])
    raise RuntimeError("No coexistence root was found")


def run(
    N=128,
    L=10.0,
    T=200.0,
    seed=0,
    noise_amp=1.0e-3,
    zero_mean_noise=False,
    vmax=None,
    d1=0.01,
    d2=0.1,
    d3=1.0,
    xi=20.0,
    sigma0=0.2,
    sigma1=0.5,
    alpha=0.1,
    gamma=1.0,
    beta=1.0,
    delta=0.5,
    ell=1.0,
    margin=None,
    ic="turing",
    cfl=0.4,
    sample_from=None,
    sample_every=5.0,
    log=None,
):
    dx = L / N
    rng = np.random.default_rng(seed)
    centres = (np.arange(N) + 0.5) * dx
    X, Y = np.meshgrid(centres, centres, indexing="ij")

    if margin is None:
        s0 = np.full((N, N), float(sigma0))
    else:
        hi, rm = margin
        distance = np.minimum.reduce([X, L - X, Y, L - Y])
        s0 = np.where(distance <= rm, hi, 0.0)

    zu = rng.standard_normal((N, N))
    zv = rng.standard_normal((N, N))
    zw = rng.standard_normal((N, N))
    if zero_mean_noise:
        zu -= zu.mean()
        zv -= zv.mean()
        zw -= zw.mean()

    if ic == "turing":
        ustar = coexistence(sigma0, sigma1, alpha, gamma, beta, delta, ell)
        u = ustar + noise_amp * zu
        v = 1.0 - ustar + noise_amp * zv
        w = (
            ustar * (alpha + gamma * (1.0 - ustar)) / ell + noise_amp * zw
        )
    elif ic == "margin":
        u = 0.5 + noise_amp * zu
        v = 0.5 + noise_amp * zv
        w = np.zeros((N, N))
    else:
        raise ValueError("ic must be 'turing' or 'margin'")
    u, v, w = (np.clip(field, 0.0, None) for field in (u, v, w))
    initial_means = dict(u=float(u.mean()), v=float(v.mean()), w=float(w.mean()))

    t = 0.0
    minimum = np.inf
    dt_previous = None
    factors = {}
    u_ref, v_ref, w_ref, t_ref = u.copy(), v.copy(), w.copy(), 0.0
    samples = []
    t_next = sample_from if sample_from is not None else np.inf
    K1max = K2max = 0.0
    step = 0
    while t < T:
        if step % 25 == 0:
            gradient_max = max(grad_mag_max(w, dx), 1.0e-12)
            dt_raw = min(cfl * dx / (xi * gradient_max + 1.0e-12), 0.02)
            dt = 0.5 ** np.ceil(-np.log2(max(dt_raw, 1.0e-6)))
        dt = min(dt, T - t)
        step += 1
        if dt != dt_previous:
            factors = {
                key: dct_diffusion_factors(N, L, diffusion, dt)
                for key, diffusion in (("u", d1), ("v", d2), ("w", d3))
            }
            dt_previous = dt

        reaction_u = u * (1.0 - u - v)
        reaction_v = (
            s0 + sigma1 * w / (1.0 + w) - delta * v - beta * u * v
        )
        reaction_w = alpha * u + gamma * u * v - ell * w
        chem = chemotaxis_div(v, w, xi, dx, vmax)
        u = implicit_diffuse(u + dt * reaction_u, factors["u"])
        v = implicit_diffuse(v + dt * (reaction_v - chem), factors["v"])
        w = implicit_diffuse(w + dt * reaction_w, factors["w"])
        if not (np.all(np.isfinite(u)) and np.all(np.isfinite(v)) and np.all(np.isfinite(w))):
            raise FloatingPointError("The two-dimensional solver produced a non-finite value")

        minimum = min(minimum, float(u.min()), float(v.min()), float(w.min()))
        t += dt
        if t > 1.0:
            K1max = max(K1max, grad_mag_max(w, dx))
            K2max = max(K2max, lap_max(w, dx))
        if t >= T - 10.0 and t_ref == 0.0:
            u_ref, v_ref, w_ref, t_ref = u.copy(), v.copy(), w.copy(), t
        if t >= t_next:
            samples.append(
                dict(
                    t=float(t),
                    K1=grad_mag_max(w, dx),
                    K2=lap_max(w, dx),
                    maxv=float(v.max()),
                    maxu=float(u.max()),
                )
            )
            t_next = t + sample_every
            if log is not None:
                with Path(log).open("a", encoding="utf-8") as handle:
                    snap = samples[-1]
                    handle.write(
                        f"t={t:10.5f} maxv={snap['maxv']:.10g} "
                        f"K1={snap['K1']:.10g} K2={snap['K2']:.10g}\n"
                    )

    drifts = dict(
        u=relative_change(u, u_ref),
        v=relative_change(v, v_ref),
        w=relative_change(w, w_ref),
    )
    drifts["max"] = max(drifts.values())
    drifts["interval"] = float(T - t_ref)
    return dict(
        u=u,
        v=v,
        w=w,
        dx=float(dx),
        s0=s0,
        s0bar=float(s0.mean()),
        mmin=float(minimum),
        K1=float(K1max),
        K2=float(K2max),
        drift=float(drifts["u"]),
        stationarity=drifts,
        samples=samples,
        steps=int(step),
        t_ref=float(t_ref),
        initial_means=initial_means,
        params=dict(
            N=N,
            L=L,
            T=T,
            seed=seed,
            noise_amp=noise_amp,
            zero_mean_noise=zero_mean_noise,
            vmax=vmax,
            d1=d1,
            d2=d2,
            d3=d3,
            xi=xi,
            sigma0=sigma0,
            sigma1=sigma1,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            delta=delta,
            ell=ell,
            margin=margin,
            ic=ic,
            cfl=cfl,
        ),
    )


def cosine(a, b):
    denominator = np.sqrt((a**2).sum()) * np.sqrt((b**2).sum())
    return float((a * b).sum() / denominator)


def biomarkers(result, threshold_u, threshold_v):
    u, v, w = result["u"], result["v"], result["w"]
    tumour = u > threshold_u
    count = tumour.sum()
    hot = float(((u > threshold_u) & (v > threshold_v)).sum() / count) if count else np.nan
    cold = float(((u > threshold_u) & (v <= threshold_v)).sum() / count) if count else np.nan
    cuv, cuw = cosine(u, v), cosine(u, w)
    return dict(H_hot=hot, H_cold=cold, I_excl=1.0 - cuv, M_chem=cuw - cuv)


def supercriticality(result, d2, d3, xi, gamma):
    area = result["dx"] ** 2
    mass_v = float(result["v"].sum() * area)
    ustar = float(np.percentile(result["u"], 90))
    critical_mass = 8.0 * np.pi * d2 * d3 / (xi * gamma * ustar)
    return mass_v, ustar, critical_mass, mass_v / critical_mass

