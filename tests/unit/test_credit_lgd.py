"""Tests for the Beta-distributed LGD."""

from __future__ import annotations

import numpy as np
import pytest

from tranche_pricing.credit.lgd import BetaLGDParams, pdf, sample


def test_params_reject_infeasible_variance() -> None:
    with pytest.raises(ValueError, match="cannot have std"):
        BetaLGDParams(mean=0.85, std=0.40)


def test_params_reject_invalid_mean() -> None:
    with pytest.raises(ValueError):
        BetaLGDParams(mean=0.0, std=0.10)
    with pytest.raises(ValueError):
        BetaLGDParams(mean=1.0, std=0.10)


def test_alpha_beta_shape_match_method_of_moments() -> None:
    p = BetaLGDParams(mean=0.7, std=0.1)
    var = p.std**2
    scale = p.mean * (1 - p.mean) / var - 1.0
    expected_alpha = p.mean * scale
    expected_beta = (1 - p.mean) * scale
    assert p.alpha_shape == pytest.approx(expected_alpha)
    assert p.beta_shape == pytest.approx(expected_beta)


def test_sample_mean_and_std_match_params() -> None:
    rng = np.random.default_rng(0)
    p = BetaLGDParams(mean=0.85, std=0.12)
    x = sample(p, n_samples=200000, rng=rng)
    assert abs(x.mean() - p.mean) < 5e-3
    assert abs(x.std(ddof=1) - p.std) < 5e-3


def test_pdf_integrates_to_one() -> None:
    p = BetaLGDParams(mean=0.85, std=0.12)
    xs = np.linspace(1e-4, 1 - 1e-4, 5000)
    integral = float(np.trapezoid(pdf(p, xs), xs))
    assert abs(integral - 1.0) < 1e-3


def test_sample_within_unit_interval() -> None:
    rng = np.random.default_rng(2)
    p = BetaLGDParams(mean=0.5, std=0.20)
    x = sample(p, n_samples=10000, rng=rng)
    assert (x > 0).all() and (x < 1).all()
