"""Tests for the ``FitResult`` dataclass."""

from __future__ import annotations

from dataclasses import dataclass

from tranche_pricing.calibration._types import FitResult


@dataclass(frozen=True, slots=True)
class _ToyParams:
    a: float
    b: float


def test_n_params_counts_dataclass_fields() -> None:
    fit = FitResult(params=_ToyParams(a=1.0, b=2.0), log_likelihood=-10.0, n_obs=100)
    assert fit.n_params == 2


def test_aic_bic_distinct_for_dict_params() -> None:
    # When ``params`` is a dict (used by ``cox_calibrate.calibrate``),
    # ``n_params`` counts the dict keys so that AIC and BIC differ.
    fit = FitResult(
        params={"kappa": 0.5, "theta": 0.08, "xi": 0.02},
        log_likelihood=-50.0,
        n_obs=200,
    )
    assert fit.n_params == 3
    assert fit.aic != fit.bic
    # AIC = 2k - 2 logL = 2*3 - 2*(-50) = 106
    assert fit.aic == 106.0


def test_n_params_unknown_type_returns_zero() -> None:
    fit = FitResult(params="opaque", log_likelihood=-1.0, n_obs=10)
    assert fit.n_params == 0
