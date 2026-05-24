"""One-factor Student-t copula for portfolio credit risk.

The construction mirrors the Gaussian copula but applies a global
chi-square mixing variable so the joint distribution has heavier tails::

    X_i = (sqrt(rho) * M + sqrt(1 - rho) * Z_i) * sqrt(nu / W),
    M, Z_i ~ N(0, 1) independent, W ~ chi2(nu) independent,

so each ``X_i`` is Student-t with ``nu`` degrees of freedom and the joint
distribution is the symmetric one-factor t-copula. Default times follow from
the constant-hazard map applied to the t-CDF marginals.

The key empirical feature is the non-zero **lower tail dependence**, which
the Gaussian copula misses by construction; see :func:`tail_dependence_lower`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import t

from ._types import default_time_from_uniform


@dataclass(frozen=True, slots=True)
class StudentTCopulaParams:
    """One-factor Student-t copula parameters: correlation and dof."""

    rho: float
    nu: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.rho < 1.0:
            raise ValueError("rho must be in [0, 1).")
        if self.nu <= 2:
            raise ValueError("Student-t nu must be > 2 (else variance undefined).")


def simulate_default_times(
    params: StudentTCopulaParams,
    *,
    horizon_years: float,
    pd_terminal: float,
    n_sims: int,
    n_obligors: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Simulate ``(n_sims, n_obligors)`` default times under the Student-t copula."""
    if n_sims <= 0 or n_obligors <= 0:
        raise ValueError("n_sims and n_obligors must be positive.")

    sqrt_rho = float(np.sqrt(params.rho))
    sqrt_1m = float(np.sqrt(1.0 - params.rho))

    m = rng.standard_normal(size=n_sims)
    z = rng.standard_normal(size=(n_sims, n_obligors))
    # Chi-square mixing variable (shared across obligors within each sim).
    w = rng.chisquare(df=params.nu, size=n_sims)
    scale = np.sqrt(params.nu / w)[:, None]

    x = (sqrt_rho * m[:, None] + sqrt_1m * z) * scale
    u = t.cdf(x, df=params.nu)
    return default_time_from_uniform(u, horizon_years=horizon_years, pd_terminal=pd_terminal)


def tail_dependence_lower(rho: float, nu: float) -> float:
    """Theoretical lower tail-dependence ``lambda_L`` for the t-copula.

    Embrechts–McNeil–Straumann (2002):

        lambda_L = 2 * T_{nu+1}( -sqrt((nu+1)(1-rho)/(1+rho)) ).

    The Gaussian copula has ``lambda_L = 0`` for any ``rho < 1``; with the
    same ``rho`` the t-copula's tail dependence grows as ``nu`` shrinks,
    making joint extreme losses materially more likely.
    """
    if not 0 <= rho < 1:
        raise ValueError("rho must be in [0, 1).")
    if nu <= 0:
        raise ValueError("nu must be positive.")
    arg = -np.sqrt((nu + 1.0) * (1.0 - rho) / (1.0 + rho))
    return float(2.0 * t.cdf(arg, df=nu + 1))


__all__ = [
    "StudentTCopulaParams",
    "simulate_default_times",
    "tail_dependence_lower",
]
