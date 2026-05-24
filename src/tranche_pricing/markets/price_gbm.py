"""Geometric Brownian motion for the property-price dynamics.

The risk-neutral / physical SDE for the building value :math:`S_t`::

    dS_t = mu * S_t * dt + sigma * S_t * dW_t

The exact-step solution

    S_{t+dt} = S_t * exp((mu - sigma^2 / 2) * dt + sigma * sqrt(dt) * Z),   Z ~ N(0,1)

avoids discretisation bias and is what we use everywhere in the Monte Carlo
engine. ``mu`` and ``sigma`` are expressed in annualised units; the data layer
delivers quarterly observations of the Notaires-INSEE index, so the
calibration in :mod:`tranche_pricing.calibration.mle_gbm` uses ``dt = 0.25``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class GBMParams:
    """Annualised drift and volatility of a geometric Brownian motion."""

    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError(f"GBM sigma must be > 0, got {self.sigma!r}.")


def simulate_paths(
    params: GBMParams,
    *,
    s0: float,
    n_paths: int,
    n_steps: int,
    dt: float,
    rng: np.random.Generator,
    antithetic: bool = False,
) -> NDArray[np.float64]:
    """Simulate paths under GBM and return an array of shape (n_paths, n_steps+1).

    Parameters
    ----------
    params
        Annualised drift and volatility.
    s0
        Initial value (S_0).
    n_paths
        Number of independent paths to simulate.
    n_steps
        Number of time steps; the returned array has ``n_steps + 1`` columns
        because the initial value is included.
    dt
        Time-step size in years (e.g. 1/12 for monthly).
    rng
        ``numpy.random.Generator`` instance (use ``np.random.default_rng(seed)``).
    antithetic
        If True, pair every Gaussian draw with its negation; effective number
        of paths is then ``n_paths`` (half antithetic).
    """
    if s0 <= 0:
        raise ValueError("s0 must be > 0.")
    if dt <= 0:
        raise ValueError("dt must be > 0.")
    if n_paths <= 0 or n_steps <= 0:
        raise ValueError("n_paths and n_steps must be positive.")

    drift = (params.mu - 0.5 * params.sigma**2) * dt
    diffusion = params.sigma * np.sqrt(dt)

    if antithetic:
        half = n_paths // 2
        z_half = rng.standard_normal(size=(half, n_steps))
        z = np.concatenate([z_half, -z_half], axis=0)
        if z.shape[0] < n_paths:  # odd n_paths: pad with one extra draw
            extra = rng.standard_normal(size=(1, n_steps))
            z = np.concatenate([z, extra], axis=0)
    else:
        z = rng.standard_normal(size=(n_paths, n_steps))

    log_increments = drift + diffusion * z
    log_path = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1)
    return np.asarray(s0 * np.exp(log_path), dtype=np.float64)


def log_return_moments(params: GBMParams, dt: float) -> tuple[float, float]:
    """Return the mean and variance of log-returns over a ``dt``-long horizon.

    This is what MLE on observed log-returns aims to recover:
    ``mean = (mu - sigma^2 / 2) * dt`` and ``var = sigma^2 * dt``.
    """
    return (params.mu - 0.5 * params.sigma**2) * dt, params.sigma**2 * dt


__all__ = ["GBMParams", "log_return_moments", "simulate_paths"]
