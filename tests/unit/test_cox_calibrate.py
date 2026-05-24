"""Smoke tests for the CIR Gaussian-transition MLE used by the Cox model."""

from __future__ import annotations

import numpy as np
import pytest

from tranche_pricing.calibration.cox_calibrate import calibrate


def _simulate_cir(
    *,
    kappa: float,
    theta: float,
    xi: float,
    y0: float,
    n: int,
    dt: float,
    seed: int = 0,
) -> np.ndarray:
    """Generate a CIR series via Euler-Maruyama with full-truncation."""
    rng = np.random.default_rng(seed)
    y = np.empty(n)
    y[0] = y0
    sqrt_dt = float(np.sqrt(dt))
    for t in range(1, n):
        z = rng.standard_normal()
        y_pos = max(y[t - 1], 0.0)
        y[t] = max(
            y[t - 1] + kappa * (theta - y[t - 1]) * dt + xi * np.sqrt(y_pos) * sqrt_dt * z, 1e-12
        )
    return y


def test_calibrate_recovers_mean_within_tolerance() -> None:
    series = _simulate_cir(kappa=0.3, theta=0.08, xi=0.05, y0=0.08, n=2000, dt=0.25)
    fit = calibrate(series, dt=0.25)
    assert fit.params["kappa"] > 0
    assert fit.params["theta"] == pytest.approx(0.08, rel=0.30)
    assert fit.params["xi"] > 0
    assert np.isfinite(fit.log_likelihood)


def test_calibrate_rejects_non_positive_series() -> None:
    series = np.array([0.01, -0.02, 0.03])
    with pytest.raises(ValueError):
        calibrate(series, dt=0.25)


def test_calibrate_feller_diagnostic_present() -> None:
    series = _simulate_cir(kappa=0.5, theta=0.10, xi=0.05, y0=0.10, n=500, dt=0.25)
    fit = calibrate(series, dt=0.25)
    assert "feller_condition" in fit.extra
    assert isinstance(fit.extra["feller_condition"], (bool, np.bool_))
