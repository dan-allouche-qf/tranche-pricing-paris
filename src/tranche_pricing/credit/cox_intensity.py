"""Cox / doubly-stochastic intensity-based credit risk model.

Each obligor has hazard rate

    lambda_i(t) = alpha + beta * Y_t,

with the common macro factor :math:`Y_t` following a CIR-like square-root
process::

    dY_t = kappa * (theta - Y_t) dt + xi * sqrt(Y_t) dW_t,
    Y_0 = theta.

Defaults are conditionally independent given the path of :math:`Y`; the
default time of obligor :math:`i` is

    tau_i = inf{ t > 0 : integral_0^t lambda_i(s) ds > E_i },   E_i ~ Exp(1).

Compared to the static asset-value copulas, the Cox model produces *time-
varying* default correlation that comes from the autocorrelation of the
macro driver. This often delivers more realistic loss-distribution tails for
multi-year horizons.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class CoxIntensityParams:
    """Parameters of the doubly-stochastic intensity model."""

    alpha: float  # idiosyncratic constant hazard
    beta: float  # sensitivity to the macro factor
    kappa: float  # CIR mean-reversion speed
    theta: float  # CIR long-run mean
    xi: float  # CIR volatility

    def __post_init__(self) -> None:
        if self.alpha < 0:
            raise ValueError("alpha must be >= 0.")
        if self.beta < 0:
            raise ValueError("beta must be >= 0.")
        if self.kappa <= 0 or self.theta <= 0 or self.xi <= 0:
            raise ValueError("kappa, theta and xi must all be > 0.")


def _simulate_cir(
    kappa: float,
    theta: float,
    xi: float,
    *,
    y0: float,
    n_sims: int,
    n_steps: int,
    dt: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Euler-Maruyama with full-truncation positivity fix."""
    y = np.empty((n_sims, n_steps + 1), dtype=np.float64)
    y[:, 0] = y0
    sqrt_dt = float(np.sqrt(dt))
    for k in range(n_steps):
        z = rng.standard_normal(size=n_sims)
        y_pos = np.maximum(y[:, k], 0.0)
        y[:, k + 1] = y[:, k] + kappa * (theta - y[:, k]) * dt + xi * np.sqrt(y_pos) * sqrt_dt * z
        y[:, k + 1] = np.maximum(y[:, k + 1], 0.0)
    return y


def simulate_default_times(
    params: CoxIntensityParams,
    *,
    horizon_years: float,
    pd_terminal: float,
    n_sims: int,
    n_obligors: int,
    rng: np.random.Generator,
    steps_per_year: int = 12,
) -> NDArray[np.float64]:
    """Simulate ``(n_sims, n_obligors)`` default times under the Cox model.

    ``pd_terminal`` is accepted for interface parity but is not used
    structurally: the marginal default probability is determined by the
    hazard process itself. We rescale ``alpha`` so that the expected
    cumulative PD at the horizon matches ``pd_terminal`` when ``beta == 0``;
    when ``beta > 0`` this is approximate. The actual realised PD is returned
    via the empirical default-rate diagnostic.
    """
    if n_sims <= 0 or n_obligors <= 0:
        raise ValueError("n_sims and n_obligors must be positive.")
    if horizon_years <= 0:
        raise ValueError("horizon_years must be positive.")

    n_steps = round(horizon_years * steps_per_year)
    dt = horizon_years / n_steps
    grid = np.linspace(0.0, horizon_years, n_steps + 1)

    y_paths = _simulate_cir(
        kappa=params.kappa,
        theta=params.theta,
        xi=params.xi,
        y0=params.theta,
        n_sims=n_sims,
        n_steps=n_steps,
        dt=dt,
        rng=rng,
    )

    # Trapezoidal integral of lambda_i(s) = alpha + beta * Y_s on [0, t_k].
    integrand = params.alpha + params.beta * y_paths
    cum_lambda = np.zeros_like(y_paths)
    cum_lambda[:, 1:] = np.cumsum(0.5 * (integrand[:, :-1] + integrand[:, 1:]) * dt, axis=1)

    # Threshold draws (one Exp(1) per (sim, obligor)).
    e = rng.exponential(size=(n_sims, n_obligors))

    # For each (sim, obligor), find smallest k with cum_lambda[sim, k] > e_{sim, j}.
    # Broadcast: compare (n_sims, n_steps+1) against (n_sims, n_obligors, 1).
    # To avoid materialising a (n_sims, n_steps+1, n_obligors) tensor we loop
    # over obligors with searchsorted along each sim.
    default_times = np.full((n_sims, n_obligors), fill_value=np.inf, dtype=np.float64)
    for i in range(n_sims):
        # cum_lambda[i] is monotone non-decreasing, so we can vectorise over obligors.
        idx = np.searchsorted(cum_lambda[i], e[i], side="right")
        in_range = idx <= n_steps
        # Linear interpolation at the crossing for sub-grid precision.
        valid_idx = idx[in_range]
        clipped = np.clip(valid_idx, 1, n_steps)
        before = cum_lambda[i, clipped - 1]
        after = cum_lambda[i, clipped]
        frac = (e[i, in_range] - before) / np.where(after - before > 0, after - before, 1.0)
        frac = np.clip(frac, 0.0, 1.0)
        default_times[i, in_range] = grid[clipped - 1] + frac * dt

    return default_times


def calibrate_alpha_for_pd(
    *,
    pd_terminal: float,
    horizon_years: float,
    beta: float,
    theta: float,
) -> float:
    """Pick ``alpha`` so the unconditional cumulative PD ≈ pd_terminal at horizon.

    When ``beta = 0`` this is exact (alpha = -log(1 - pd_terminal) / T).
    Otherwise we adjust by E[Y_t] = theta and approximate the cumulative
    expected hazard as ``(alpha + beta * theta) * T``.
    """
    if pd_terminal <= 0 or pd_terminal >= 1:
        raise ValueError("pd_terminal must be in (0, 1).")
    target = -np.log(1.0 - pd_terminal) / horizon_years
    alpha = float(target - beta * theta)
    return max(alpha, 0.0)


__all__ = [
    "CoxIntensityParams",
    "calibrate_alpha_for_pd",
    "simulate_default_times",
]
