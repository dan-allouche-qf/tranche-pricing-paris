"""Merton (1976) jump-diffusion for property prices.

The SDE generalises geometric Brownian motion with a compound-Poisson jump
component:

    dS_t / S_t = (mu - lambda * kappa) dt + sigma dW_t + (J - 1) dN_t

with

    N_t ~ Poisson(lambda)
    log J ~ N(mu_J, sigma_J^2)
    kappa = E[J - 1] = exp(mu_J + sigma_J^2 / 2) - 1   (martingale correction)

Over a horizon dt the log-return is

    r = (mu - sigma^2 / 2 - lambda * kappa) dt + sigma * sqrt(dt) * Z + sum_{i=1}^N log J_i,

a mixture: conditional on ``N = k`` jumps over ``dt``, ``r`` is normal with mean

    m_k = (mu - sigma^2 / 2 - lambda * kappa) dt + k * mu_J

and variance

    s2_k = sigma^2 * dt + k * sigma_J^2.

The mixture weights are ``P(N = k) = exp(-lambda dt) (lambda dt)^k / k!``.
Density and MLE truncate the sum at ``k_max`` (default 20), which is more
than enough for plausible Paris real-estate values of ``lambda dt``.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import factorial

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


@dataclass(frozen=True, slots=True)
class MertonParams:
    """Annualised Merton (1976) jump-diffusion parameters."""

    mu: float
    sigma: float
    lam: float  # jump intensity (jumps per year)
    mu_jump: float  # mean of log J
    sigma_jump: float

    def __post_init__(self) -> None:
        if self.sigma <= 0 or self.sigma_jump <= 0 or self.lam < 0:
            raise ValueError(
                f"Invalid Merton parameters: sigma={self.sigma}, "
                f"sigma_jump={self.sigma_jump}, lam={self.lam}."
            )

    @property
    def kappa(self) -> float:
        """Mean jump size minus one: ``E[J - 1] = exp(mu_J + sigma_J^2 / 2) - 1``."""
        return float(np.exp(self.mu_jump + 0.5 * self.sigma_jump**2) - 1.0)


def simulate_paths(
    params: MertonParams,
    *,
    s0: float,
    n_paths: int,
    n_steps: int,
    dt: float,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Simulate paths under Merton jump-diffusion; shape (n_paths, n_steps+1)."""
    if s0 <= 0 or dt <= 0 or n_paths <= 0 or n_steps <= 0:
        raise ValueError("s0, dt, n_paths and n_steps must all be positive.")

    z = rng.standard_normal(size=(n_paths, n_steps))
    n_jumps = rng.poisson(lam=params.lam * dt, size=(n_paths, n_steps))
    diffusion = (params.mu - 0.5 * params.sigma**2 - params.lam * params.kappa) * dt + (
        params.sigma * np.sqrt(dt) * z
    )

    # Vectorised aggregation of log J_i for each (path, step):
    # given k jumps, sum_{i=1}^k log J_i ~ N(k * mu_J, k * sigma_J^2).
    mean_jump = n_jumps * params.mu_jump
    std_jump = np.sqrt(n_jumps) * params.sigma_jump
    jump_part = mean_jump + std_jump * rng.standard_normal(size=n_jumps.shape)

    log_increments = diffusion + jump_part
    log_path = np.concatenate([np.zeros((n_paths, 1)), np.cumsum(log_increments, axis=1)], axis=1)
    return np.asarray(s0 * np.exp(log_path), dtype=np.float64)


def log_density(
    r: NDArray[np.float64],
    params: MertonParams,
    *,
    dt: float,
    k_max: int = 20,
) -> NDArray[np.float64]:
    """Log-density of one-step log-returns under the Merton mixture.

    Returns the log-density evaluated at each element of ``r``. Uses the
    truncated Poisson-mixture representation; ``k_max`` controls the
    truncation horizon.
    """
    if dt <= 0 or k_max < 0:
        raise ValueError("Require dt > 0 and k_max >= 0.")
    lam_dt = params.lam * dt
    weights = np.exp(-lam_dt) * np.array([lam_dt**k / factorial(k) for k in range(k_max + 1)])

    drift = (params.mu - 0.5 * params.sigma**2 - params.lam * params.kappa) * dt
    mixture_pdf = np.zeros_like(r, dtype=float)
    for k in range(k_max + 1):
        mean = drift + k * params.mu_jump
        var = params.sigma**2 * dt + k * params.sigma_jump**2
        std = np.sqrt(var)
        mixture_pdf += weights[k] * norm.pdf(r, loc=mean, scale=std)

    # Floor the density at a tiny positive number to keep logs finite.
    return np.asarray(np.log(np.maximum(mixture_pdf, 1e-300)), dtype=np.float64)


__all__ = ["MertonParams", "log_density", "simulate_paths"]
