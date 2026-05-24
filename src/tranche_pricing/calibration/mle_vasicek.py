"""MLE for the Vasicek short-rate model on equally-spaced observations.

Under Vasicek the transition is Gaussian with closed-form conditional mean
and variance::

    r_{t+dt} | r_t ~ N(b + (r_t - b) e^{-a dt},  sigma_r^2 / (2 a) * (1 - e^{-2 a dt}))

Equivalently this is an AR(1) regression of ``r_{t+dt}`` on ``r_t``::

    r_{t+dt} = alpha + beta * r_t + epsilon_t,
    epsilon_t ~ N(0, sigma_eps^2),
    beta = e^{-a dt},
    alpha = b * (1 - beta),
    sigma_eps^2 = sigma_r^2 / (2 a) * (1 - beta^2).

OLS gives the exact MLE here. We invert the mapping to recover
:math:`(a, b, \\sigma_r)` and report analytical standard errors from the
Gaussian Fisher information.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from ..markets.rates_vasicek import VasicekParams
from ._types import FitResult


def calibrate(
    rates: NDArray[np.float64],
    *,
    dt: float,
) -> FitResult[VasicekParams]:
    """Fit Vasicek by exact MLE on equally-spaced short-rate observations.

    Parameters
    ----------
    rates
        Vector of ``r_t`` observations (in decimal, i.e. 0.03 = 3 %).
    dt
        Time step between consecutive observations, in years.
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")
    r = np.asarray(rates, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 5:
        raise ValueError("Need at least 5 finite observations.")

    r_t = r[:-1]
    r_t1 = r[1:]
    n_obs = r_t.size

    # OLS regression of r_{t+dt} on r_t.
    x_mean = float(r_t.mean())
    y_mean = float(r_t1.mean())
    cov = float(np.mean((r_t - x_mean) * (r_t1 - y_mean)) * n_obs / (n_obs - 1))
    var_x = float(r_t.var(ddof=1))
    beta = cov / var_x
    alpha = y_mean - beta * x_mean
    residuals = r_t1 - (alpha + beta * r_t)
    sigma_eps2 = float(residuals.var(ddof=2))  # 2 parameters: alpha, beta

    # We need |beta| < 1 for mean reversion (a > 0). Floor / clip to a tiny
    # safety margin to avoid log(0) when the data are nearly random-walk.
    if beta >= 1.0:
        beta = 1.0 - 1e-6
    elif beta <= 0.0:
        beta = 1e-6
    a = -np.log(beta) / dt
    b = alpha / (1.0 - beta)
    sigma_r = float(np.sqrt(2.0 * a * sigma_eps2 / (1.0 - beta**2)))

    # Log-likelihood of the AR(1) representation.
    ll = float(norm.logpdf(residuals, loc=0.0, scale=np.sqrt(sigma_eps2)).sum())

    # Analytical Fisher-information SEs (OLS Gaussian) translated through the
    # parameter mapping a = -log(beta) / dt, b = alpha / (1 - beta),
    # sigma_r^2 = 2 a sigma_eps^2 / (1 - beta^2).
    sum_x2 = float(((r_t - x_mean) ** 2).sum())
    se_beta = float(np.sqrt(sigma_eps2 / sum_x2))
    se_alpha = float(np.sqrt(sigma_eps2 * (1.0 / n_obs + x_mean**2 / sum_x2)))
    # Delta method:
    se_a = float(se_beta / (beta * dt))
    se_b = float(
        np.sqrt((se_alpha / (1.0 - beta)) ** 2 + (alpha * se_beta / (1.0 - beta) ** 2) ** 2)
    )
    se_sigma_eps = float(np.sqrt(sigma_eps2 / (2.0 * (n_obs - 2))))
    se_sigma_r = float(
        sigma_r
        * np.sqrt(
            (se_sigma_eps / np.sqrt(sigma_eps2)) ** 2
            + 0.5 * (se_a / a) ** 2
            + (beta * se_beta / (1.0 - beta**2)) ** 2
        )
    )

    return FitResult(
        params=VasicekParams(a=a, b=b, sigma_r=sigma_r),
        std_errors={"a": se_a, "b": se_b, "sigma_r": se_sigma_r},
        log_likelihood=ll,
        n_obs=n_obs,
        extra={
            "dt": dt,
            "alpha": alpha,
            "beta": beta,
            "sigma_eps": float(np.sqrt(sigma_eps2)),
            "half_life_years": float(np.log(2.0) / a),
        },
    )


__all__ = ["calibrate"]
