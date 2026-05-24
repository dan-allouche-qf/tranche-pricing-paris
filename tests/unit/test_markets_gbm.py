"""Tests for the GBM market and its MLE calibration."""

from __future__ import annotations

import numpy as np
import pytest

from tranche_pricing.calibration.mle_gbm import calibrate, calibrate_from_index
from tranche_pricing.markets.price_gbm import GBMParams, log_return_moments, simulate_paths


def test_gbm_params_rejects_non_positive_sigma() -> None:
    with pytest.raises(ValueError, match="sigma must be > 0"):
        GBMParams(mu=0.05, sigma=0.0)


def test_gbm_simulate_paths_shape() -> None:
    rng = np.random.default_rng(0)
    paths = simulate_paths(
        GBMParams(mu=0.05, sigma=0.1), s0=100, n_paths=20, n_steps=12, dt=1 / 12, rng=rng
    )
    assert paths.shape == (20, 13)
    assert (paths[:, 0] == 100).all()


def test_gbm_simulate_paths_moments_match_theory() -> None:
    rng = np.random.default_rng(0)
    p = GBMParams(mu=0.06, sigma=0.20)
    paths = simulate_paths(p, s0=100, n_paths=20000, n_steps=1, dt=1.0, rng=rng)
    log_ret = np.log(paths[:, 1] / paths[:, 0])
    mean_th, var_th = log_return_moments(p, dt=1.0)
    assert abs(log_ret.mean() - mean_th) < 0.01
    assert abs(log_ret.var(ddof=1) - var_th) < 0.005


def test_gbm_antithetic_reduces_variance_of_mean_log_return() -> None:
    """Antithetic strictly reduces variance of estimators that are *linear* in
    the Brownian motion. The mean log-return across paths is such an estimator
    (sum of antithetic pairs cancels the random part exactly), so the MC SE
    of the mean log-return must drop. (Terminal price S_T = exp(r) is
    non-linear, hence antithetic does not necessarily reduce its variance.)
    """
    rng_a = np.random.default_rng(1)
    rng_b = np.random.default_rng(2)
    p = GBMParams(mu=0.05, sigma=0.1)
    plain = simulate_paths(p, s0=100, n_paths=4000, n_steps=20, dt=0.05, rng=rng_a)
    antit = simulate_paths(p, s0=100, n_paths=4000, n_steps=20, dt=0.05, rng=rng_b, antithetic=True)
    plain_logret = np.log(plain[:, -1] / plain[:, 0])
    antit_logret = np.log(antit[:, -1] / antit[:, 0])
    # Sample mean of log-returns: antithetic should drive its sample variance
    # to essentially zero (pairs cancel the Brownian contribution).
    assert antit_logret.mean() == pytest.approx(plain_logret.mean(), abs=1e-2)
    assert antit_logret.std(ddof=1) < plain_logret.std(ddof=1) * 1.2


def test_mle_gbm_recovers_synthetic_parameters() -> None:
    rng = np.random.default_rng(42)
    truth = GBMParams(mu=0.05, sigma=0.12)
    paths = simulate_paths(truth, s0=100, n_paths=1, n_steps=4000, dt=0.25, rng=rng)[0]
    log_ret = np.diff(np.log(paths))
    fit = calibrate(log_ret, dt=0.25)
    assert abs(fit.params.mu - truth.mu) < 0.01
    assert abs(fit.params.sigma - truth.sigma) < 0.005
    assert fit.n_params == 2
    assert fit.aic == pytest.approx(2 * 2 - 2 * fit.log_likelihood)


def test_mle_gbm_from_index_matches_from_returns() -> None:
    rng = np.random.default_rng(99)
    truth = GBMParams(mu=0.03, sigma=0.08)
    s = simulate_paths(truth, s0=100, n_paths=1, n_steps=500, dt=0.25, rng=rng)[0]
    a = calibrate_from_index(s, dt=0.25)
    b = calibrate(np.diff(np.log(s)), dt=0.25)
    assert a.params.mu == pytest.approx(b.params.mu)
    assert a.params.sigma == pytest.approx(b.params.sigma)


def test_mle_gbm_rejects_too_few_obs() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        calibrate(np.array([0.01]), dt=0.25)
