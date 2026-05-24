"""Tests for the vectorised risk-metric helpers."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import norm

from tranche_pricing.risk.metrics import (
    calmar_ratio,
    expected_shortfall,
    max_drawdown,
    omega_ratio,
    sharpe_ratio,
    sortino_ratio,
    summary,
    tracking_error,
    value_at_risk,
)


def test_var_matches_gaussian_quantile() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(loc=0.05, scale=0.10, size=200000)
    var95 = value_at_risk(x, alpha=0.95)
    expected = -float(norm.ppf(0.05, loc=0.05, scale=0.10))
    assert abs(var95 - expected) < 0.005


def test_es_at_least_var() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=50000)
    var = value_at_risk(x, alpha=0.95)
    es = expected_shortfall(x, alpha=0.95)
    assert es >= var


def test_sharpe_on_riskless_returns_diverges() -> None:
    x = np.full(200, 0.05)
    val = sharpe_ratio(x, rf=0.03)
    # On a constant series the empirical std is float-precision noise, so
    # the ratio explodes — either to NaN or to a very large value.
    assert np.isnan(val) or abs(val) > 1e6


def test_sortino_zero_when_no_downside() -> None:
    x = np.full(50, 0.05)
    val = sortino_ratio(x, target=0.03)
    # All returns above target → no downside → either inf (positive excess) or nan.
    assert val == float("inf") or np.isnan(val)


def test_sortino_uses_full_sample_mean_for_downside_dev() -> None:
    # Construct returns where downside-only mean and full-sample mean differ
    # materially: 80 upside paths at +0.10, 20 downside paths at -0.20.
    upside = np.full(80, 0.10)
    downside = np.full(20, -0.20)
    returns = np.concatenate([upside, downside])
    val = sortino_ratio(returns, target=0.0)
    # Canonical Sortino-van der Meer (1991): the downside denominator is
    # sqrt(E[(target - r)_+^2]), with the expectation taken over the full
    # sample rather than the downside subset alone.
    expected_dd = float(np.sqrt(((-downside) ** 2).sum() / returns.size))
    expected = float(returns.mean()) / expected_dd
    assert val == pytest.approx(expected, rel=1e-9)
    # Sanity: a downside-only mean would shrink the denominator and
    # inflate the ratio.
    downside_only_dd = float(np.sqrt((downside**2).mean()))
    assert expected_dd < downside_only_dd


def test_max_drawdown_simple_case() -> None:
    cash = np.array([1.0, 1.10, 1.20, 0.90, 1.00, 1.30])
    # Peak 1.20, trough 0.90 → drawdown = 0.30 / 1.20 = 0.25.
    assert max_drawdown(cash) == pytest.approx(0.25)


def test_omega_above_one_when_returns_skewed_positive() -> None:
    rng = np.random.default_rng(0)
    x = rng.lognormal(mean=0.05, sigma=0.30, size=20000) - 1.0
    assert omega_ratio(x, threshold=0.0) > 1.0


def test_omega_below_one_when_returns_skewed_negative() -> None:
    rng = np.random.default_rng(0)
    x = -rng.lognormal(mean=0.05, sigma=0.30, size=20000) + 0.5
    assert omega_ratio(x, threshold=0.0) < 1.0


def test_calmar_uses_mean_return_and_drawdown() -> None:
    cash = np.array([1.0, 1.1, 0.8, 1.0])
    returns = np.array([0.1, -0.3, 0.25])
    calmar = calmar_ratio(returns, cash)
    expected = float(returns.mean()) / max_drawdown(cash)
    assert calmar == pytest.approx(expected)


def test_tracking_error_zero_when_identical() -> None:
    x = np.array([0.01, 0.02, 0.03, 0.04])
    assert tracking_error(x, x) == pytest.approx(0.0)


def test_summary_returns_expected_keys() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=300)
    out = summary(returns=x, rf=0.0)
    for key in ("mean", "std", "sharpe", "sortino", "omega", "var_95", "var_99", "es_95", "es_99"):
        assert key in out
