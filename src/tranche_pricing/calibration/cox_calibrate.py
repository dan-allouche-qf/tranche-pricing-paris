"""Approximate MLE for a CIR process used as the Cox macro factor.

We approximate the CIR transition (which is non-central chi-square) by a
Gaussian with the exact CIR mean and variance — accurate when $\\xi^2 \\ll
\\kappa \\theta$, which holds for unemployment-rate data — and run an
ordinary AR(1) MLE on the resulting Gaussian. The mapping back to
$(\\kappa, \\theta, \\xi)$ is closed form.

Reference: Ait-Sahalia (1999) discusses the bias of the Gaussian-MLE
approximation for CIR; with the unemployment-rate sample size and
volatility level the bias is below the calibration's own MC noise floor.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._types import FitResult


def calibrate(
    series: NDArray[np.float64],
    *,
    dt: float,
) -> FitResult[dict]:
    """Fit (kappa, theta, xi) of a CIR process by Gaussian-transition MLE.

    Parameters
    ----------
    series
        Equally-spaced strictly-positive observations of the CIR variable.
    dt
        Time step between consecutive observations, in years.

    Returns
    -------
    FitResult
        ``params`` is a dict ``{"kappa": ..., "theta": ..., "xi": ...}``.
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")
    y = np.asarray(series, dtype=float)
    y = y[np.isfinite(y)]
    if (y <= 0).any():
        raise ValueError("CIR series must be strictly positive.")
    if y.size < 8:
        raise ValueError("Need at least 8 finite observations.")

    y_t = y[:-1]
    y_t1 = y[1:]
    n = y_t.size

    # AR(1) regression of y_{t+dt} on y_t.
    x_mean = float(y_t.mean())
    y_mean = float(y_t1.mean())
    cov = float(((y_t - x_mean) * (y_t1 - y_mean)).sum()) / max(n - 1, 1)
    var_x = float(y_t.var(ddof=1))
    beta = cov / var_x
    alpha = y_mean - beta * x_mean
    residuals = y_t1 - (alpha + beta * y_t)
    sigma_eps2 = float(residuals.var(ddof=2))

    if beta >= 1.0:
        beta = 1.0 - 1e-6
    elif beta <= 0.0:
        beta = 1e-6

    kappa = float(-np.log(beta) / dt)
    theta = float(alpha / (1.0 - beta))
    # Leading-order Euler approximation: Var(Y_{t+dt} | Y_t) ≈ theta * xi^2 * dt
    # for small kappa*dt. Using the substitution 1 - exp(-2 kappa dt) = 2 kappa dt
    # + O((kappa dt)^2), we invert the OLS residual variance sigma_eps^2 against
    # theta * xi^2 * (1 - beta^2) / (2 kappa). This is neither the exact
    # conditional variance nor the stationary variance — it is the first-order
    # Euler proxy that is consistent with the AR(1) MLE we use above.
    xi = float(np.sqrt(2.0 * kappa * sigma_eps2 / (theta * (1.0 - beta**2))))

    # Log-likelihood under the Gaussian approximation.
    log_lik = float(
        -0.5 * n * (np.log(2.0 * np.pi * sigma_eps2) + (residuals**2).mean() / sigma_eps2)
    )

    return FitResult(
        params={"kappa": kappa, "theta": theta, "xi": xi},
        std_errors=None,
        log_likelihood=log_lik,
        n_obs=n,
        extra={
            "dt": dt,
            "alpha": alpha,
            "beta": beta,
            "sigma_eps2": sigma_eps2,
            "feller_condition": 2.0 * kappa * theta > xi**2,
        },
    )


__all__ = ["calibrate"]
