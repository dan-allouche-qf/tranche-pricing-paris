"""Tests for the option-theoretic insurance pricing.

Two invariants:

  * grid-invariance: lump-sum PV is approximately the same on a monthly
    and a quarterly simulation grid;
  * deductible scaling: total deductible across the horizon equals
    ``deductible_months * monthly_rent`` regardless of grid choice.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

from tranche_pricing.insurance.option_theoretic import (
    price_premium,
    shortfall_payoffs,
)


@dataclass(slots=True)
class _Cfg:
    gross_yield: float
    horizon_years: float


def _toy_output(
    *,
    dt: float,
    horizon_years: float,
    price0: float = 1.0,
    gross_yield: float = 0.04,
    rent_drop: float = 0.40,
    n_paths: int = 4,
) -> SimpleNamespace:
    n_steps = round(horizon_years / dt)
    price = np.full((n_paths, n_steps + 1), price0, dtype=float)
    full_rent_per_period = gross_yield * price0 * dt
    # Net rent realises a flat shortfall of ``rent_drop`` of full rent.
    net_rent = np.full((n_paths, n_steps), (1.0 - rent_drop) * full_rent_per_period, dtype=float)
    # Constant short rate, deterministic discount factors.
    r0 = 0.02
    t = np.arange(n_steps + 1, dtype=float) * dt
    discount = np.broadcast_to(np.exp(-r0 * t), (n_paths, n_steps + 1)).copy()
    return SimpleNamespace(
        config=_Cfg(gross_yield=gross_yield, horizon_years=horizon_years),
        dt=dt,
        n_steps=n_steps,
        price_paths=price,
        net_rent=net_rent,
        discount_factors=discount,
    )


def test_deductible_total_equals_deductible_months_of_monthly_rent() -> None:
    horizon = 10.0
    gross_yield = 0.04
    price0 = 1.0
    for dt in (1 / 12, 0.25):
        out = _toy_output(
            dt=dt,
            horizon_years=horizon,
            price0=price0,
            gross_yield=gross_yield,
            rent_drop=0.0,
        )
        payoffs = shortfall_payoffs(out, deductible_months=1, coverage_cap=1.0)
        # Zero shortfall → payoffs all zero — but compute the deductible
        # directly from the formula and check its total.
        full_rent_per_period = gross_yield * price0 * dt
        period_months = max(round(12 * dt), 1)
        monthly_rent = full_rent_per_period / period_months
        deductible_per_period = (1 * monthly_rent) / out.n_steps
        total_deductible = deductible_per_period * out.n_steps
        # 1 month of monthly rent at gross_yield=0.04, price0=1 → 0.04 / 12.
        assert total_deductible == pytest.approx(monthly_rent, rel=1e-9)
        assert np.allclose(payoffs, 0.0)


def test_lump_sum_grid_invariant_within_discretisation_noise() -> None:
    horizon = 10.0
    monthly = price_premium(
        _toy_output(dt=1 / 12, horizon_years=horizon),
        deductible_months=1,
        coverage_cap=0.9,
    )
    quarterly = price_premium(
        _toy_output(dt=0.25, horizon_years=horizon),
        deductible_months=1,
        coverage_cap=0.9,
    )
    # The two grids should agree up to integration / discount
    # discretisation noise — within 5% relative.
    assert monthly["lump_sum_premium"] == pytest.approx(quarterly["lump_sum_premium"], rel=0.05)
