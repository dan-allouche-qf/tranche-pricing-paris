"""Tests for the one-factor Student-t copula and its tail-dependence formula."""

from __future__ import annotations

import numpy as np
import pytest

from tranche_pricing.credit.student_t_copula import (
    StudentTCopulaParams,
    simulate_default_times,
    tail_dependence_lower,
)


def test_params_rejects_low_nu() -> None:
    with pytest.raises(ValueError):
        StudentTCopulaParams(rho=0.2, nu=1.5)


def test_shape_and_pd_match_target() -> None:
    rng = np.random.default_rng(0)
    tau = simulate_default_times(
        StudentTCopulaParams(rho=0.15, nu=5.0),
        horizon_years=10.0,
        pd_terminal=0.20,
        n_sims=4000,
        n_obligors=100,
        rng=rng,
    )
    assert tau.shape == (4000, 100)
    empirical_pd = (tau <= 10.0).mean()
    assert abs(empirical_pd - 0.20) < 0.01


def test_tail_dependence_formula_matches_simulation() -> None:
    """Empirical Pr(U_j < q | U_i < q) at small q approximates lambda_L."""
    rng = np.random.default_rng(2)
    rho, nu = 0.30, 3.0
    n_sims, n_obl = 20000, 4
    tau = simulate_default_times(
        StudentTCopulaParams(rho=rho, nu=nu),
        horizon_years=10.0,
        pd_terminal=0.05,
        n_sims=n_sims,
        n_obligors=n_obl,
        rng=rng,
    )
    # Two obligors: lower-tail dependence => high empirical pairwise default prob.
    both_default = ((tau[:, 0] <= 10.0) & (tau[:, 1] <= 10.0)).mean()
    single_default = (tau[:, 0] <= 10.0).mean()
    empirical_lambda = both_default / single_default if single_default > 0 else 0.0
    theoretical_lambda = tail_dependence_lower(rho=rho, nu=nu)
    # Tolerance: this is a tail quantity, MC noise is real.
    assert empirical_lambda > 0.05
    assert abs(empirical_lambda - theoretical_lambda) < 0.10


def test_tail_dependence_zero_for_independent_marginals() -> None:
    """rho = 0 still gives positive tail dependence under t-copula (asymptotic)."""
    # At rho = 0 and small nu the tail dependence is positive (unlike Gaussian).
    val_t = tail_dependence_lower(rho=0.0, nu=3.0)
    assert val_t > 0


def test_tail_dependence_monotone_in_rho() -> None:
    rho_grid = [0.0, 0.1, 0.3, 0.6]
    vals = [tail_dependence_lower(rho=r, nu=4.0) for r in rho_grid]
    assert vals == sorted(vals)


def test_tail_dependence_increasing_as_nu_decreases() -> None:
    nu_grid = [30.0, 8.0, 4.0, 3.0]
    vals = [tail_dependence_lower(rho=0.30, nu=nu) for nu in nu_grid]
    assert vals == sorted(vals)
