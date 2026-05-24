"""MLE for the GBM property-price model.

Closed-form: given i.i.d. log-returns :math:`r_i = \\log(S_{t_i}/S_{t_{i-1}})`
with horizon ``dt`` (in years), the log-likelihood is the sum of a normal
density at mean :math:`(\\mu - \\sigma^2/2)\\,dt` and variance :math:`\\sigma^2\\,dt`.
Equating the score to zero gives

    sigma_hat^2 = Var(r) / dt
    mu_hat = mean(r) / dt + sigma_hat^2 / 2

Standard errors follow from the inverse Fisher information of the normal
density (with the chain-rule adjustment :math:`\\mu = (\\bar r + \\sigma^2 dt / 2) / dt`).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm

from ..markets.price_gbm import GBMParams
from ._types import FitResult


def calibrate(
    log_returns: NDArray[np.float64],
    *,
    dt: float,
) -> FitResult[GBMParams]:
    """Fit a GBM by closed-form MLE on a vector of equally-spaced log-returns.

    Parameters
    ----------
    log_returns
        Vector :math:`r_i = \\log(S_{t_i}/S_{t_{i-1}})`. NaNs are dropped.
    dt
        Time step in years (e.g. ``0.25`` for quarterly data).
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")
    r = np.asarray(log_returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 3:
        raise ValueError("Need at least 3 finite log-returns to fit GBM.")

    mean_r = float(r.mean())
    var_r = float(r.var(ddof=1))
    sigma_hat = float(np.sqrt(var_r / dt))
    mu_hat = float(mean_r / dt + 0.5 * sigma_hat**2)

    # Log-likelihood at the optimum: sum of N(mean_r, var_r) log-density.
    ll = float(norm.logpdf(r, loc=mean_r, scale=np.sqrt(var_r)).sum())

    # Standard errors from the inverse Fisher information of the normal:
    # SE(mean) = sigma_hat * sqrt(dt) / sqrt(n);  SE(sigma) = sigma_hat / sqrt(2 (n - 1)).
    se_sigma = sigma_hat / np.sqrt(2.0 * (n - 1))
    se_mu = sigma_hat * np.sqrt(dt) / np.sqrt(n) / dt  # delta-method on mu = mean / dt + sigma^2/2
    se_mu = float(np.sqrt(se_mu**2 + (sigma_hat * se_sigma) ** 2))

    return FitResult(
        params=GBMParams(mu=mu_hat, sigma=sigma_hat),
        std_errors={"mu": float(se_mu), "sigma": float(se_sigma)},
        log_likelihood=ll,
        n_obs=n,
        extra={"dt": dt, "mean_log_return": mean_r, "var_log_return": var_r},
    )


def calibrate_from_index(
    index: NDArray[np.float64],
    *,
    dt: float,
) -> FitResult[GBMParams]:
    """Convenience: fit GBM from a price-level series instead of log-returns."""
    s = np.asarray(index, dtype=float)
    s = s[np.isfinite(s)]
    if s.size < 4:
        raise ValueError("Need at least 4 finite price observations.")
    r = np.diff(np.log(s))
    return calibrate(r, dt=dt)


__all__ = ["calibrate", "calibrate_from_index"]
