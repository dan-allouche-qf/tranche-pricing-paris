"""Tests for the block bootstrap helper."""

from __future__ import annotations

import numpy as np
import pytest

from tranche_pricing.calibration.bootstrap import block_bootstrap_indices, bootstrap


def test_block_bootstrap_indices_shape_and_bounds() -> None:
    rng = np.random.default_rng(0)
    idx = block_bootstrap_indices(n=120, block_size=10, n_resamples=50, rng=rng)
    assert idx.shape == (50, 120)
    assert idx.min() >= 0
    assert idx.max() < 120


def test_block_bootstrap_rejects_invalid_inputs() -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        block_bootstrap_indices(n=0, block_size=10, n_resamples=10, rng=rng)
    with pytest.raises(ValueError):
        block_bootstrap_indices(n=10, block_size=0, n_resamples=10, rng=rng)


def test_bootstrap_recovers_mean_and_std_of_normal_sample() -> None:
    rng = np.random.default_rng(7)
    series = rng.normal(loc=2.0, scale=3.0, size=500)
    result = bootstrap(
        series,
        fit_fn=lambda x: {"mean": float(x.mean()), "std": float(x.std(ddof=1))},
        n_resamples=400,
        block_size=1,
        seed=42,
    )
    assert "mean" in result.samples and "std" in result.samples
    assert abs(np.mean(result.samples["mean"]) - 2.0) < 0.5
    # For a 500-obs sample, SE(mean) ~ sigma / sqrt(n) ~ 0.13. Use 0.4 as loose UB.
    assert result.std_errors()["mean"] < 0.4


def test_bootstrap_quantiles_are_ordered() -> None:
    rng = np.random.default_rng(1)
    series = rng.normal(size=200)
    result = bootstrap(
        series,
        fit_fn=lambda x: {"mean": float(x.mean())},
        n_resamples=200,
    )
    lo, hi = result.quantiles(alpha=0.05)["mean"]
    assert lo < hi
