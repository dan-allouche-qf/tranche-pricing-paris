"""Vasicek (1977) short-rate model.

Risk-neutral / physical SDE for the instantaneous short rate ``r_t``::

    dr_t = a * (b - r_t) * dt + sigma_r * dW_t

The transition law is Gaussian with closed form::

    r_{t+dt} | r_t ~ N( b + (r_t - b) * exp(-a * dt),
                        sigma_r^2 / (2 a) * (1 - exp(-2 a dt)) )

which is what the MLE in :mod:`tranche_pricing.calibration.mle_vasicek` uses
(no Euler discretisation bias). The zero-coupon bond price under Vasicek has
a clean closed form too, which we use as a discount-factor sanity check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class VasicekParams:
    """Vasicek short-rate parameters."""

    a: float  # mean-reversion speed
    b: float  # long-run mean
    sigma_r: float  # instantaneous volatility

    def __post_init__(self) -> None:
        if self.a <= 0:
            raise ValueError("Vasicek mean-reversion speed a must be > 0.")
        if self.sigma_r <= 0:
            raise ValueError("Vasicek sigma_r must be > 0.")


def simulate_paths(
    params: VasicekParams,
    *,
    r0: float,
    n_paths: int,
    n_steps: int,
    dt: float,
    rng: np.random.Generator,
    antithetic: bool = False,
) -> NDArray[np.float64]:
    """Simulate paths of ``r_t`` using the exact Gaussian transition."""
    if dt <= 0 or n_paths <= 0 or n_steps <= 0:
        raise ValueError("dt, n_paths and n_steps must be positive.")

    a, b, sigma_r = params.a, params.b, params.sigma_r
    exp_term = np.exp(-a * dt)
    cond_var = (sigma_r**2) * (1.0 - np.exp(-2.0 * a * dt)) / (2.0 * a)
    cond_std = np.sqrt(cond_var)

    if antithetic:
        half = n_paths // 2
        z_half = rng.standard_normal(size=(half, n_steps))
        z = np.concatenate([z_half, -z_half], axis=0)
        if z.shape[0] < n_paths:
            extra = rng.standard_normal(size=(1, n_steps))
            z = np.concatenate([z, extra], axis=0)
    else:
        z = rng.standard_normal(size=(n_paths, n_steps))

    paths = np.empty((n_paths, n_steps + 1), dtype=np.float64)
    paths[:, 0] = r0
    for t in range(n_steps):
        paths[:, t + 1] = b + (paths[:, t] - b) * exp_term + cond_std * z[:, t]
    return paths


def stationary_moments(params: VasicekParams) -> tuple[float, float]:
    """Long-run mean and variance of ``r_t`` under Vasicek.

    The stationary distribution is ``N(b, sigma_r^2 / (2 a))``.
    """
    return params.b, params.sigma_r**2 / (2.0 * params.a)


def zero_coupon_bond_price(
    params: VasicekParams,
    *,
    r: float | NDArray[np.float64],
    maturity: float,
) -> float | NDArray[np.float64]:
    """Closed-form Vasicek zero-coupon bond price ``P(t, t + maturity | r_t = r)``.

    Reference: Vasicek (1977), eq. (28). Used to discount terminal cash flows
    and as a sanity test against simulated discount factors.
    """
    if maturity < 0:
        raise ValueError("maturity must be non-negative.")
    a, b, sigma_r = params.a, params.b, params.sigma_r
    if maturity == 0:
        return r * 0.0 + 1.0 if isinstance(r, np.ndarray) else 1.0

    bcal = (1.0 - np.exp(-a * maturity)) / a
    long_term = b - (sigma_r**2) / (2.0 * a**2)
    acal_log = long_term * (bcal - maturity) - (sigma_r**2) * bcal**2 / (4.0 * a)
    if isinstance(r, np.ndarray):
        return np.asarray(np.exp(acal_log - bcal * r), dtype=np.float64)
    return float(np.exp(acal_log - bcal * r))


__all__ = [
    "VasicekParams",
    "simulate_paths",
    "stationary_moments",
    "zero_coupon_bond_price",
]
