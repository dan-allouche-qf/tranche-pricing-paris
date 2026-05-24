"""One-factor Gaussian copula for portfolio credit risk.

This is the Li (2000) / Vasicek (2002) model: each obligor's asset value
:math:`X_i` is a linear combination of a single common factor :math:`M` and
an idiosyncratic shock :math:`Z_i`, both standard normal::

    X_i = sqrt(rho) * M + sqrt(1 - rho) * Z_i,    M, Z_i ~ N(0,1),

so that ``corr(X_i, X_j) = rho`` for ``i != j`` and each ``X_i`` has standard
normal marginal. Under exponential survival with constant hazard, default
times are obtained by ``tau_i = -log(1 - Phi(X_i)) / h`` (capped at ``inf``).

Reference: Li (2000), Vasicek (1987, 2002). The large-portfolio limit
expression for the conditional cumulative loss is implemented in
:func:`large_portfolio_loss`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from ._types import constant_hazard, default_time_from_uniform


@dataclass(frozen=True, slots=True)
class GaussianCopulaParams:
    """One-factor Gaussian copula correlation parameter."""

    rho: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.rho < 1.0:
            raise ValueError("rho must be in [0, 1).")


def simulate_default_times(
    params: GaussianCopulaParams,
    *,
    horizon_years: float,
    pd_terminal: float,
    n_sims: int,
    n_obligors: int,
    rng: np.random.Generator,
    antithetic: bool = False,
) -> NDArray[np.float64]:
    """Simulate ``(n_sims, n_obligors)`` default times under the Gaussian copula."""
    if n_sims <= 0 or n_obligors <= 0:
        raise ValueError("n_sims and n_obligors must be positive.")
    rho = params.rho
    sqrt_rho = float(np.sqrt(rho))
    sqrt_1m = float(np.sqrt(1.0 - rho))

    if antithetic:
        half = n_sims // 2
        m_half = rng.standard_normal(size=half)
        m = np.concatenate([m_half, -m_half])
        if m.size < n_sims:
            m = np.concatenate([m, rng.standard_normal(size=1)])
        z_half = rng.standard_normal(size=(half, n_obligors))
        z = np.concatenate([z_half, -z_half], axis=0)
        if z.shape[0] < n_sims:
            z = np.concatenate([z, rng.standard_normal(size=(1, n_obligors))], axis=0)
    else:
        m = rng.standard_normal(size=n_sims)
        z = rng.standard_normal(size=(n_sims, n_obligors))

    x = sqrt_rho * m[:, None] + sqrt_1m * z
    u = norm.cdf(x)
    return default_time_from_uniform(u, horizon_years=horizon_years, pd_terminal=pd_terminal)


def large_portfolio_loss(
    *,
    rho: float,
    pd_terminal: float,
    horizon_years: float,
    quantile: float,
) -> float:
    """Vasicek (2002) closed-form quantile of the cumulative portfolio loss.

    In the large-portfolio limit (``n_obligors -> infty``) the cumulative
    portfolio loss at the terminal horizon is

        L = Phi( (Phi^{-1}(p) - sqrt(rho) * Phi^{-1}(1 - q)) / sqrt(1 - rho) )

    where ``p`` is the unconditional cumulative PD and ``q`` is the desired
    confidence level (e.g. 0.99 for the 99th percentile of losses).
    """
    if not 0 < quantile < 1:
        raise ValueError("quantile must be in (0, 1).")
    if not 0 <= rho < 1:
        raise ValueError("rho must be in [0, 1).")
    p = 1.0 - float(np.exp(-constant_hazard(pd_terminal, horizon_years) * horizon_years))
    return float(
        norm.cdf((norm.ppf(p) - np.sqrt(rho) * norm.ppf(1.0 - quantile)) / np.sqrt(1.0 - rho))
    )


def conditional_pd(m: float, *, rho: float, pd_terminal: float) -> float:
    """Conditional default probability given the common factor ``M = m``."""
    return float(norm.cdf((norm.ppf(pd_terminal) - np.sqrt(rho) * m) / np.sqrt(1.0 - rho)))


__all__ = [
    "GaussianCopulaParams",
    "conditional_pd",
    "large_portfolio_loss",
    "simulate_default_times",
]
