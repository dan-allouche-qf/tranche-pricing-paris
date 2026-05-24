"""Numerical MLE for the Merton (1976) jump-diffusion log-return density.

Given i.i.d. log-returns :math:`r_i` of horizon ``dt`` the log-likelihood is
the sum of the Poisson-mixture log-density :func:`tranche_pricing.markets.
price_jump.log_density`. We optimise it with ``scipy.optimize.minimize`` and a
constrained parameterisation (positive ``sigma``, ``sigma_jump``, ``lam``)
through softplus / exp transforms.

The likelihood surface is multi-modal — a single L-BFGS-B start from a
moments-based seed routinely collapses to the degenerate ``sigma_jump = 0``
solution on short samples. We therefore run ``n_starts`` perturbed
initialisations and keep the best non-collapsed log-likelihood. When every
restart collapses we return the best collapsing fit and flag
``extra["collapsed"] = True`` so the caller (and the working paper)
documents the identification failure honestly rather than reporting a
spurious zero-variance jump.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from ..markets.price_jump import MertonParams, log_density
from ._types import FitResult

# A Merton fit with sigma_jump below this floor is treated as a collapsed
# solution where the jump component has been absorbed into the diffusive
# part.
_SIGMA_JUMP_COLLAPSE_FLOOR: float = 1e-3


def _from_unconstrained(theta: NDArray[np.float64]) -> MertonParams:
    """Map an unconstrained parameter vector to a valid ``MertonParams``."""
    mu, log_sigma, log_lam, mu_jump, log_sigma_jump = theta
    return MertonParams(
        mu=float(mu),
        sigma=float(np.exp(log_sigma)),
        lam=float(np.exp(log_lam)),
        mu_jump=float(mu_jump),
        sigma_jump=float(np.exp(log_sigma_jump)),
    )


def _to_unconstrained(params: MertonParams) -> NDArray[np.float64]:
    return np.array(
        [
            params.mu,
            np.log(params.sigma),
            np.log(max(params.lam, 1e-6)),
            params.mu_jump,
            np.log(params.sigma_jump),
        ],
        dtype=float,
    )


def _moments_seed(r: NDArray[np.float64], dt: float) -> MertonParams:
    """Sensible starting point anchored on the GBM moment estimators."""
    sigma_init = float(np.std(r, ddof=1) / np.sqrt(dt))
    return MertonParams(
        mu=float(np.mean(r) / dt + 0.5 * sigma_init**2),
        sigma=sigma_init,
        lam=0.5,
        mu_jump=-0.02,
        sigma_jump=0.05,
    )


def _perturbed_seed(base: MertonParams, rng: np.random.Generator) -> MertonParams:
    """Random restart drawn around the moments seed to escape local minima.

    The diffusive parameters ``mu`` and ``sigma`` stay anchored to the GBM
    fit; only the three jump parameters are randomised because that is
    where the multi-modality lives.
    """
    return MertonParams(
        mu=base.mu,
        sigma=base.sigma,
        lam=float(np.clip(np.exp(rng.normal(np.log(0.5), 0.6)), 0.05, 5.0)),
        mu_jump=float(rng.normal(-0.02, 0.05)),
        sigma_jump=float(np.clip(np.exp(rng.normal(np.log(0.05), 0.6)), 0.01, 0.30)),
    )


def calibrate(
    log_returns: NDArray[np.float64],
    *,
    dt: float,
    init: MertonParams | None = None,
    k_max: int = 20,
    maxiter: int = 400,
    n_starts: int = 10,
    start_seed: int = 20260519,
) -> FitResult[MertonParams]:
    """Fit Merton jump-diffusion by multi-start MLE on equally-spaced log-returns.

    Parameters
    ----------
    log_returns
        Equally-spaced log-return vector. NaNs are dropped.
    dt
        Time step between consecutive observations, in years.
    init
        Optional starting point for the first restart. Defaults to a moments
        seed anchored on the GBM MLE.
    k_max
        Truncation horizon for the Poisson mixture in the likelihood.
    maxiter
        Maximum iterations of L-BFGS-B per restart.
    n_starts
        Number of restarts. Restart 0 uses ``init`` (or the moments seed if
        ``init`` is ``None``); the remaining restarts perturb the jump
        component randomly via :func:`_perturbed_seed`.
    start_seed
        Master seed for the perturbations.
    """
    if dt <= 0:
        raise ValueError("dt must be positive.")
    if n_starts < 1:
        raise ValueError("n_starts must be at least 1.")
    r = np.asarray(log_returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 8:
        raise ValueError("Need at least 8 finite log-returns to fit Merton.")

    base_seed = init if init is not None else _moments_seed(r, dt)
    rng = np.random.default_rng(start_seed)

    def neg_log_likelihood(theta: NDArray[np.float64]) -> float:
        try:
            params = _from_unconstrained(theta)
        except ValueError:
            return 1e10
        try:
            ll_val = float(log_density(r, params, dt=dt, k_max=k_max).sum())
        except (ValueError, FloatingPointError):
            return 1e10
        return -ll_val

    seeds = [base_seed]
    for _ in range(n_starts - 1):
        seeds.append(_perturbed_seed(base_seed, rng))

    fits: list[tuple[float, MertonParams, bool, int]] = []  # (ll, params, collapsed, n_iter)
    for seed in seeds:
        theta0 = _to_unconstrained(seed)
        result = minimize(
            neg_log_likelihood,
            theta0,
            method="L-BFGS-B",
            options={"maxiter": maxiter, "ftol": 1e-9},
        )
        params = _from_unconstrained(result.x)
        ll_at_x = float(-result.fun)
        collapsed = params.sigma_jump < _SIGMA_JUMP_COLLAPSE_FLOOR
        fits.append((ll_at_x, params, collapsed, int(result.nit)))

    # Prefer non-collapsed fits sorted by LL; fall back to the best collapsed.
    non_collapsed = [f for f in fits if not f[2]]
    if non_collapsed:
        non_collapsed.sort(key=lambda x: x[0], reverse=True)
        ll, params, collapsed, n_iter = non_collapsed[0]
    else:
        fits.sort(key=lambda x: x[0], reverse=True)
        ll, params, collapsed, n_iter = fits[0]

    return FitResult(
        params=params,
        std_errors=None,  # numerical Hessian inversion is fragile; use bootstrap.
        log_likelihood=ll,
        n_obs=n,
        extra={
            "dt": dt,
            "n_starts": int(n_starts),
            "n_iter_best": n_iter,
            "k_max": int(k_max),
            "init_params": base_seed,
            "collapsed": bool(collapsed),
            "n_non_collapsed_starts": len(non_collapsed),
            "ll_per_start": [float(f[0]) for f in fits],
        },
    )


__all__ = ["calibrate"]
