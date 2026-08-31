"""One-dimensional finite-volume solver and manuscript diagnostics."""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import csr_matrix, diags, kron as sp_kron


def grid(L: float, N: int) -> tuple[float, np.ndarray]:
    dx = L / N
    return dx, (np.arange(N) + 0.5) * dx


def margin_profile(x: np.ndarray, L: float, hi: float, rm: float) -> np.ndarray:
    """Cell-centre source rule used by all time-dependent manuscript runs."""
    return np.where(np.minimum(x, L - x) <= rm, hi, 0.0)


def cell_average_margin_profile(L: float, N: int, hi: float, rm: float) -> np.ndarray:
    """Exact cell averages of a two-sided discontinuous margin source.

    This rule is used only by the threshold-convergence calculation, exactly as
    stated in Example 1.  It removes jumps caused by moving a cell centre across
    the thin-source boundary while bisecting in L.
    """
    dx = L / N
    left = np.arange(N) * dx
    right = left + dx
    overlap_left = np.maximum(0.0, np.minimum(right, rm) - left)
    overlap_right = np.maximum(
        0.0, np.minimum(right, L) - np.maximum(left, L - rm)
    )
    return hi * (overlap_left + overlap_right) / dx


def rhs(t: float, y: np.ndarray, p: dict) -> np.ndarray:
    N, dx = p["N"], p["dx"]
    u, v, w = y[:N], y[N : 2 * N], y[2 * N :]

    def lap(z: np.ndarray, diffusion: float) -> np.ndarray:
        flux = np.zeros(N + 1)
        flux[1:-1] = -diffusion * (z[1:] - z[:-1]) / dx
        return -(flux[1:] - flux[:-1]) / dx

    xi, vmax = p["xi"], p.get("vmax")
    q = (
        v
        if vmax is None
        else np.clip(v, 0.0, None) * np.clip(1.0 - v / vmax, 0.0, None)
    )
    chem_flux = np.zeros(N + 1)
    dw = (w[1:] - w[:-1]) / dx
    chem_flux[1:-1] = xi * np.where(dw >= 0.0, q[:-1], q[1:]) * dw
    chem = (chem_flux[1:] - chem_flux[:-1]) / dx

    du = lap(u, p["d1"]) + u * (1.0 - u - v)
    dv = (
        lap(v, p["d2"])
        - chem
        + p["s0"]
        + p["sigma1"] * w / (1.0 + w)
        - p["delta"] * v
        - p["beta"] * u * v
    )
    dw_dt = (
        lap(w, p["d3"])
        + p["alpha"] * u
        + p["gamma"] * u * v
        - p["ell"] * w
    )
    return np.concatenate([du, dv, dw_dt])


def jacobian_pattern(N: int) -> csr_matrix:
    tri = diags(
        [np.ones(N - 1), np.ones(N), np.ones(N - 1)],
        [-1, 0, 1],
        format="csr",
    )
    return csr_matrix(sp_kron(np.ones((3, 3)), tri, format="csr"))


def relative_change(new: np.ndarray, old: np.ndarray) -> float:
    return float(np.linalg.norm(new - old) / max(np.linalg.norm(new), 1.0e-30))


def stationarity_from_history(result: dict) -> dict[str, float]:
    """Compare the last two of the 60 stored one-dimensional snapshots."""
    drifts = {
        "u": relative_change(result["U"][:, -1], result["U"][:, -2]),
        "v": relative_change(result["V"][:, -1], result["V"][:, -2]),
        "w": relative_change(result["W"][:, -1], result["W"][:, -2]),
    }
    drifts["max"] = max(drifts.values())
    drifts["interval"] = float(result["t"][-1] - result["t"][-2])
    return drifts


def run1d(
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
    smooth_ic=None,
    fail_on_error=True,
):
    dx, x = grid(L, N)
    s0 = (
        np.full(N, sigma0, dtype=float)
        if sigma0 is not None
        else margin_profile(x, L, sigma0_hi, rm)
    )
    p = dict(
        N=N,
        dx=dx,
        d1=d1,
        d2=d2,
        d3=d3,
        xi=xi,
        s0=s0,
        sigma1=sigma1,
        alpha=alpha,
        gamma=gamma,
        beta=beta,
        delta=delta,
        ell=ell,
        vmax=vmax,
    )

    rng = np.random.default_rng(seed)
    if smooth_ic is not None:
        us, amp, kmode = smooth_ic
        ws = us * (alpha + gamma * (1.0 - us)) / ell
        y0 = np.concatenate(
            [
                us + amp * np.cos(kmode * np.pi * x / L),
                np.full(N, 1.0 - us),
                np.full(N, ws),
            ]
        )
    else:
        y0 = np.concatenate(
            [
                np.full(N, u0) + noise * rng.standard_normal(N),
                np.full(N, v0),
                np.full(N, w0),
            ]
        )
    y0 = np.clip(y0, 0.0, None)

    output_times = np.linspace(0.0, T, nsamp)
    sol = solve_ivp(
        rhs,
        (0.0, T),
        y0,
        args=(p,),
        method="BDF",
        t_eval=output_times,
        jac_sparsity=jacobian_pattern(N),
        rtol=rtol,
        atol=atol,
    )
    if fail_on_error and not sol.success:
        raise RuntimeError(f"BDF integration failed: {sol.message}")
    if not np.all(np.isfinite(sol.y)):
        raise FloatingPointError("BDF integration produced a non-finite value")

    U, V, W = sol.y[:N], sol.y[N : 2 * N], sol.y[2 * N :]
    Wx = np.gradient(W, dx, axis=0)
    Wxx = np.gradient(Wx, dx, axis=0)
    K1, K2 = float(np.abs(Wx).max()), float(np.abs(Wxx).max())
    Mv = float(V.max())
    kappa = (-xi * K1 + np.sqrt((xi * K1) ** 2 + 4.0 * d2 * delta)) / (
        2.0 * d2
    )
    a = L / 2.0 - rm
    access = 1.0 / (kappa * a) if kappa > 0.0 else np.inf
    u, v, w = U[:, -1], V[:, -1], W[:, -1]
    interior = (x > rm) & (x < L - rm)
    result = dict(
        x=x,
        dx=dx,
        u=u,
        v=v,
        w=w,
        t=sol.t,
        U=U,
        V=V,
        W=W,
        ok=bool(sol.success),
        solver_message=sol.message,
        K1=K1,
        K2=K2,
        Mv=Mv,
        kappa=float(kappa),
        A=float(access),
        a=float(a),
        s0bar=float(s0.mean()),
        s0=s0,
        mass_int=float(u[interior].sum() * dx),
        mean_int=float(u[interior].mean()),
        mass_margin=float(u[~interior].sum() * dx),
        vmean_int=float(v[interior].mean()),
        mmin=float(min(U.min(), V.min(), W.min())),
        Mv_L1=float(v.sum() * dx),
        nfev=int(sol.nfev),
        njev=int(sol.njev),
        nlu=int(sol.nlu),
        params=dict(
            L=L,
            N=N,
            T=T,
            d1=d1,
            d2=d2,
            d3=d3,
            xi=xi,
            sigma0=sigma0,
            sigma0_hi=sigma0_hi,
            rm=rm,
            sigma1=sigma1,
            alpha=alpha,
            gamma=gamma,
            beta=beta,
            delta=delta,
            ell=ell,
            vmax=vmax,
            u0=u0,
            v0=v0,
            w0=w0,
            seed=seed,
            noise=noise,
            nsamp=nsamp,
            rtol=rtol,
            atol=atol,
        ),
    )
    result["stationarity"] = stationarity_from_history(result)
    return result


def weighted_L1(result: dict, eps_frac=0.5, delta=0.5, L=10.0):
    x, dx, v = result["x"], result["dx"], result["v"]
    x0, radius = L / 2.0, L / 2.0
    eps = eps_frac * delta
    params = result["params"]
    d2, xi, K1 = params["d2"], params["xi"], result["K1"]
    kappa_eps = (
        -xi * K1 + np.sqrt((xi * K1) ** 2 + 4.0 * d2 * (delta - eps))
    ) / (2.0 * d2)
    phi = np.exp(-kappa_eps * np.abs(x - x0))
    lhs = float((v * phi).sum() * dx)
    margin_length = float((result["s0"] > 0.0).sum() * dx)
    a = result["a"]
    rhs_value = (1.0 / eps) * (
        float(result["s0"].max()) * margin_length * np.exp(-kappa_eps * a)
        + params["sigma1"] * 2.0 / kappa_eps
        + d2
        * kappa_eps
        * result["Mv"]
        * 2.0
        * np.exp(-kappa_eps * radius)
    )
    return dict(
        kappa_eps=float(kappa_eps),
        lhs=lhs,
        rhs=float(rhs_value),
        ratio=float(rhs_value / lhs),
    )


def ode_comparator(
    s0bar,
    u0=0.5,
    v0=0.5,
    w0=0.0,
    T=400.0,
    sigma1=0.5,
    alpha=0.5,
    gamma=0.5,
    beta=1.0,
    delta=0.5,
    ell=1.0,
):
    def ode_rhs(t, state):
        u, v, w = state
        return [
            u * (1.0 - u - v),
            s0bar + sigma1 * w / (1.0 + w) - delta * v - beta * u * v,
            alpha * u + gamma * u * v - ell * w,
        ]

    sol = solve_ivp(
        ode_rhs,
        (0.0, T),
        [u0, v0, w0],
        method="BDF",
        rtol=1.0e-10,
        atol=1.0e-12,
    )
    if not sol.success:
        raise RuntimeError(f"ODE comparator failed: {sol.message}")
    return dict(u=float(sol.y[0, -1]), v=float(sol.y[1, -1]), w=float(sol.y[2, -1]))

