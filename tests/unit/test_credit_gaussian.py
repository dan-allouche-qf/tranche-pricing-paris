"""Tests for the one-factor Gaussian copula."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import kstest

from tranche_pricing.credit.gaussian_copula import (
    GaussianCopulaParams,
    conditional_pd,
    large_portfolio_loss,
    simulate_default_times,
)


def test_params_rejects_invalid_rho() -> None:
    with pytest.raises(ValueError):
        GaussianCopulaParams(rho=1.0)
    with pytest.raises(ValueError):
        GaussianCopulaParams(rho=-0.1)


def test_simulate_default_times_shape_and_dtype() -> None:
    rng = np.random.default_rng(0)
    tau = simulate_default_times(
        GaussianCopulaParams(rho=0.15),
        horizon_years=10.0,
        pd_terminal=0.30,
        n_sims=1000,
        n_obligors=70,
        rng=rng,
    )
    assert tau.shape == (1000, 70)
    assert tau.dtype == np.float64


def test_empirical_default_rate_matches_terminal_pd() -> None:
    rng = np.random.default_rng(7)
    n_sims, n_obl = 4000, 100
    pd_terminal = 0.20
    tau = simulate_default_times(
        GaussianCopulaParams(rho=0.20),
        horizon_years=10.0,
        pd_terminal=pd_terminal,
        n_sims=n_sims,
        n_obligors=n_obl,
        rng=rng,
    )
    empirical_pd = (tau <= 10.0).mean()
    assert abs(empirical_pd - pd_terminal) < 0.01


def test_default_rate_increasing_in_rho_does_not_hold_but_dispersion_does() -> None:
    """rho does not affect the marginal PD but does inflate cross-sim variance."""
    rng = np.random.default_rng(11)
    base = simulate_default_times(
        GaussianCopulaParams(rho=0.0),
        horizon_years=10.0,
        pd_terminal=0.10,
        n_sims=3000,
        n_obligors=100,
        rng=rng,
    )
    rng2 = np.random.default_rng(11)
    corr = simulate_default_times(
        GaussianCopulaParams(rho=0.40),
        horizon_years=10.0,
        pd_terminal=0.10,
        n_sims=3000,
        n_obligors=100,
        rng=rng2,
    )
    base_counts = (base <= 10.0).sum(axis=1)
    corr_counts = (corr <= 10.0).sum(axis=1)
    assert corr_counts.var(ddof=1) > base_counts.var(ddof=1) * 2.0


def test_uniform_marginal_after_inverse_cdf_transform() -> None:
    """Under independent ``X_i ~ N(0,1)``, ``Phi(X_i)`` is uniform [0, 1]."""
    rng = np.random.default_rng(42)
    tau = simulate_default_times(
        GaussianCopulaParams(rho=0.0),
        horizon_years=10.0,
        pd_terminal=0.999,
        n_sims=1,
        n_obligors=8000,
        rng=rng,
    )
    # tau implicit uniform = 1 - exp(-h * tau); since pd_terminal is high,
    # essentially all obligors default and tau distribution covers (0, 10].
    u_implicit = 1.0 - np.exp(-(-np.log(1 - 0.999) / 10.0) * tau[0])
    u_implicit = u_implicit[np.isfinite(u_implicit)]
    p_value = kstest(u_implicit, "uniform").pvalue
    assert p_value > 0.01


def test_conditional_pd_monotone_in_factor() -> None:
    # When the common factor M is very negative, conditional PD increases.
    high = conditional_pd(-3.0, rho=0.3, pd_terminal=0.05)
    low = conditional_pd(3.0, rho=0.3, pd_terminal=0.05)
    assert high > low > 0


def test_large_portfolio_loss_matches_vasicek_formula() -> None:
    # Numerical sanity: with rho close to 0 the 99% loss should be close to PD.
    near_zero = large_portfolio_loss(rho=1e-6, pd_terminal=0.10, horizon_years=10.0, quantile=0.99)
    assert abs(near_zero - 0.10) < 1e-3
    # With high rho the 99% loss should approach 1.0.
    high_rho = large_portfolio_loss(rho=0.99, pd_terminal=0.10, horizon_years=10.0, quantile=0.99)
    assert high_rho > 0.5
