"""Tests for the Vasicek short rate, its MLE and its analytic ZCB price."""

from __future__ import annotations

import numpy as np
import pytest

from tranche_pricing.calibration.mle_vasicek import calibrate
from tranche_pricing.markets.rates_vasicek import (
    VasicekParams,
    simulate_paths,
    stationary_moments,
    zero_coupon_bond_price,
)


def test_vasicek_params_rejects_non_positive_a() -> None:
    with pytest.raises(ValueError, match="speed a"):
        VasicekParams(a=0.0, b=0.03, sigma_r=0.01)


def test_vasicek_paths_shape_and_initial() -> None:
    rng = np.random.default_rng(0)
    p = VasicekParams(a=0.2, b=0.02, sigma_r=0.01)
    paths = simulate_paths(p, r0=0.03, n_paths=10, n_steps=12, dt=1 / 12, rng=rng)
    assert paths.shape == (10, 13)
    assert (paths[:, 0] == 0.03).all()


def test_vasicek_stationary_distribution() -> None:
    rng = np.random.default_rng(7)
    p = VasicekParams(a=2.0, b=0.025, sigma_r=0.01)
    # Long enough horizon to forget r0.
    paths = simulate_paths(p, r0=0.10, n_paths=5000, n_steps=2000, dt=1 / 52, rng=rng)
    tail = paths[:, -1]
    mean_th, var_th = stationary_moments(p)
    assert abs(tail.mean() - mean_th) < 1e-3
    assert abs(tail.var(ddof=1) - var_th) < 1e-5


def test_vasicek_zcb_at_zero_maturity_is_one() -> None:
    p = VasicekParams(a=0.2, b=0.03, sigma_r=0.01)
    assert zero_coupon_bond_price(p, r=0.03, maturity=0.0) == pytest.approx(1.0)


def test_vasicek_zcb_high_rate_below_unit() -> None:
    p = VasicekParams(a=0.2, b=0.03, sigma_r=0.01)
    price_at_zero = zero_coupon_bond_price(p, r=0.0, maturity=10.0)
    price_at_high = zero_coupon_bond_price(p, r=0.10, maturity=10.0)
    assert 0.0 < price_at_high < price_at_zero
    assert price_at_zero < 1.0  # positive long-run mean drags it below par


def test_vasicek_zcb_matches_simulated_discount_factor() -> None:
    rng = np.random.default_rng(11)
    p = VasicekParams(a=0.5, b=0.03, sigma_r=0.01)
    horizon = 5.0
    dt = 1 / 252
    n_steps = int(horizon / dt)
    paths = simulate_paths(p, r0=0.02, n_paths=20000, n_steps=n_steps, dt=dt, rng=rng)
    # Trapezoidal integral of r over [0, horizon]
    integral = 0.5 * dt * (paths[:, :-1] + paths[:, 1:]).sum(axis=1)
    mc_price = float(np.exp(-integral).mean())
    closed_form = float(zero_coupon_bond_price(p, r=0.02, maturity=horizon))
    assert abs(mc_price - closed_form) < 5e-3  # MC noise tolerance


def test_mle_vasicek_recovers_on_large_sample() -> None:
    rng = np.random.default_rng(20260519)
    truth = VasicekParams(a=0.5, b=0.025, sigma_r=0.012)
    rates = simulate_paths(truth, r0=0.025, n_paths=1, n_steps=5000, dt=1 / 52, rng=rng)[0]
    fit = calibrate(rates, dt=1 / 52)
    # Vasicek MLE has known small-sample bias on a, so we test broad tolerance.
    assert 0.3 < fit.params.a < 0.8
    assert abs(fit.params.b - truth.b) < 5e-3
    assert abs(fit.params.sigma_r - truth.sigma_r) < 2e-3
    assert fit.n_params == 3
    assert fit.std_errors is not None
    assert all(v > 0 for v in fit.std_errors.values())


def test_mle_vasicek_rejects_too_few_obs() -> None:
    with pytest.raises(ValueError, match="at least 5"):
        calibrate(np.array([0.01, 0.02]), dt=1 / 12)
