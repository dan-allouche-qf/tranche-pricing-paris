"""Tests for the Merton jump-diffusion model and its MLE."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from tranche_pricing.calibration.mle_jump import calibrate
from tranche_pricing.markets.price_jump import MertonParams, log_density, simulate_paths


def test_merton_params_validation() -> None:
    with pytest.raises(ValueError):
        MertonParams(mu=0.05, sigma=0.0, lam=0.1, mu_jump=-0.02, sigma_jump=0.05)
    with pytest.raises(ValueError):
        MertonParams(mu=0.05, sigma=0.10, lam=-1.0, mu_jump=-0.02, sigma_jump=0.05)


def test_merton_kappa_matches_definition() -> None:
    p = MertonParams(mu=0.05, sigma=0.10, lam=0.4, mu_jump=-0.03, sigma_jump=0.05)
    expected = float(np.exp(p.mu_jump + 0.5 * p.sigma_jump**2) - 1.0)
    assert p.kappa == pytest.approx(expected)


def test_merton_simulate_shape_and_initial() -> None:
    rng = np.random.default_rng(0)
    p = MertonParams(mu=0.05, sigma=0.10, lam=0.4, mu_jump=-0.03, sigma_jump=0.05)
    paths = simulate_paths(p, s0=100, n_paths=8, n_steps=12, dt=1 / 12, rng=rng)
    assert paths.shape == (8, 13)
    assert (paths[:, 0] == 100).all()


def test_merton_density_integrates_to_one() -> None:
    p = MertonParams(mu=0.04, sigma=0.10, lam=0.5, mu_jump=-0.02, sigma_jump=0.05)
    dt = 0.25
    integral, _ = quad(
        lambda r: float(np.exp(log_density(np.array([r]), p, dt=dt, k_max=20)[0])),
        -2.0,
        2.0,
        limit=200,
    )
    assert abs(integral - 1.0) < 5e-3


def test_merton_density_pure_gbm_special_case() -> None:
    """With lam = 0 the mixture collapses to a single Gaussian; check the moments."""
    p = MertonParams(mu=0.05, sigma=0.10, lam=1e-12, mu_jump=0.0, sigma_jump=0.05)
    rng = np.random.default_rng(0)
    paths = simulate_paths(p, s0=100, n_paths=10000, n_steps=1, dt=1.0, rng=rng)
    r = np.log(paths[:, 1] / paths[:, 0])
    # Theoretical: r ~ N((mu - sigma^2/2) * 1, sigma^2)
    expected_mean = p.mu - 0.5 * p.sigma**2
    assert abs(r.mean() - expected_mean) < 0.01
    assert abs(r.var(ddof=1) - p.sigma**2) < 0.005


def test_mle_jump_recovers_with_large_sample() -> None:
    rng = np.random.default_rng(20260519)
    truth = MertonParams(mu=0.04, sigma=0.07, lam=0.4, mu_jump=-0.03, sigma_jump=0.05)
    paths = simulate_paths(truth, s0=100, n_paths=1, n_steps=5000, dt=0.25, rng=rng)[0]
    r = np.diff(np.log(paths))
    fit = calibrate(r, dt=0.25)
    # Jump-diffusion identifiability is delicate even with 5000 obs because
    # lam and sigma_jump trade off (a larger lam paired with a smaller
    # sigma_jump fits a similar log-likelihood). We check sensible bounds
    # and that the multi-start picked a non-collapsed solution.
    assert abs(fit.params.mu - truth.mu) < 0.05
    assert abs(fit.params.sigma - truth.sigma) < 0.03
    assert 0.05 < fit.params.lam < 3.0
    assert 0.01 < fit.params.sigma_jump < 0.15
    assert not fit.extra["collapsed"]
    assert fit.extra["n_starts"] == 10
    assert fit.n_params == 5


def test_mle_jump_multi_start_picks_non_collapsed_over_collapsed() -> None:
    """Multi-start should find the non-collapsed optimum on a 500-obs sample.

    On a small but informative sample, a single L-BFGS-B start frequently
    converges to a degenerate sigma_jump near 0 — we verify that with 10
    restarts and a fixed seed the algorithm escapes that local minimum.
    """
    rng = np.random.default_rng(42)
    truth = MertonParams(mu=0.04, sigma=0.06, lam=0.6, mu_jump=-0.04, sigma_jump=0.07)
    paths = simulate_paths(truth, s0=100, n_paths=1, n_steps=500, dt=0.25, rng=rng)[0]
    r = np.diff(np.log(paths))

    multi = calibrate(r, dt=0.25, n_starts=10)
    assert multi.extra["n_starts"] == 10
    assert multi.extra["n_non_collapsed_starts"] >= 1
    assert not multi.extra["collapsed"]
    assert multi.params.sigma_jump > 1e-3
