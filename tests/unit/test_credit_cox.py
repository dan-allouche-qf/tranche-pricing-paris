"""Tests for the Cox doubly-stochastic intensity credit model."""

from __future__ import annotations

import numpy as np
import pytest

from tranche_pricing.credit.cox_intensity import (
    CoxIntensityParams,
    calibrate_alpha_for_pd,
    simulate_default_times,
)


def test_params_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError):
        CoxIntensityParams(alpha=-0.1, beta=1.0, kappa=0.5, theta=0.02, xi=0.05)
    with pytest.raises(ValueError):
        CoxIntensityParams(alpha=0.01, beta=1.0, kappa=0.0, theta=0.02, xi=0.05)


def test_default_times_are_within_horizon_or_inf() -> None:
    rng = np.random.default_rng(0)
    tau = simulate_default_times(
        CoxIntensityParams(alpha=0.005, beta=0.5, kappa=0.3, theta=0.03, xi=0.05),
        horizon_years=10.0,
        pd_terminal=0.15,
        n_sims=2000,
        n_obligors=50,
        rng=rng,
    )
    finite = tau[np.isfinite(tau)]
    assert finite.size > 0
    assert (finite >= 0).all()
    assert (finite <= 10.0).all()


def test_calibrate_alpha_for_zero_beta_matches_constant_hazard() -> None:
    alpha = calibrate_alpha_for_pd(pd_terminal=0.20, horizon_years=10.0, beta=0.0, theta=0.03)
    expected = -np.log(1.0 - 0.20) / 10.0
    assert alpha == pytest.approx(expected, rel=1e-6)


def test_calibrate_alpha_when_beta_positive_returns_non_negative() -> None:
    alpha = calibrate_alpha_for_pd(pd_terminal=0.20, horizon_years=10.0, beta=1.0, theta=0.03)
    assert alpha >= 0


def test_higher_xi_inflates_default_count_dispersion() -> None:
    """Higher CIR volatility xi should produce more dispersed loss distributions."""
    rng = np.random.default_rng(11)
    low_xi = simulate_default_times(
        CoxIntensityParams(alpha=0.005, beta=0.6, kappa=0.5, theta=0.03, xi=0.01),
        horizon_years=10.0,
        pd_terminal=0.10,
        n_sims=3000,
        n_obligors=100,
        rng=rng,
    )
    rng2 = np.random.default_rng(11)
    high_xi = simulate_default_times(
        CoxIntensityParams(alpha=0.005, beta=0.6, kappa=0.5, theta=0.03, xi=0.10),
        horizon_years=10.0,
        pd_terminal=0.10,
        n_sims=3000,
        n_obligors=100,
        rng=rng2,
    )
    low_var = (low_xi <= 10.0).sum(axis=1).var(ddof=1)
    high_var = (high_xi <= 10.0).sum(axis=1).var(ddof=1)
    assert high_var > low_var
