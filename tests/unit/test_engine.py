"""Tests for the Monte Carlo engine."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tranche_pricing.credit.lgd import BetaLGDParams
from tranche_pricing.markets.price_gbm import GBMParams
from tranche_pricing.markets.rates_vasicek import VasicekParams
from tranche_pricing.simulation import SimulationConfig, run_simulation, seeds
from tranche_pricing.simulation.engine import _gaussian_copula_qmc
from tranche_pricing.waterfall.tranches import Tranche


@pytest.fixture()
def baseline_inputs() -> tuple[
    SimulationConfig, GBMParams, VasicekParams, BetaLGDParams, list[Tranche], dict[str, float]
]:
    cfg = SimulationConfig(
        n_obligors=50,
        horizon_years=5.0,
        steps_per_year=12,
        par=1_000_000.0,
        gross_yield=0.04,
        maintenance_pct=0.01,
        initial_rate=0.03,
        pd_terminal=0.15,
        credit_model="gaussian_copula",
        rho=0.20,
        n_paths=600,
        master_seed=12345,
    )
    return (
        cfg,
        GBMParams(mu=0.02, sigma=0.05),
        VasicekParams(a=0.20, b=0.025, sigma_r=0.01),
        BetaLGDParams(mean=0.80, std=0.15),
        [
            Tranche("equity", 0.0, 0.25),
            Tranche("mezzanine", 0.25, 0.60),
            Tranche("senior", 0.60, 1.0),
        ],
        {"senior": 0.03, "mezzanine": 0.06, "equity": 0.0},
    )


def test_engine_output_shapes(baseline_inputs: tuple) -> None:
    cfg, gbm, vas, lgd_p, tranches, coupons = baseline_inputs
    out = run_simulation(cfg, gbm=gbm, vasicek=vas, lgd=lgd_p, tranches=tranches, coupons=coupons)
    n_steps = int(cfg.horizon_years * cfg.steps_per_year)
    assert out.price_paths.shape == (cfg.n_paths, n_steps + 1)
    assert out.rate_paths.shape == (cfg.n_paths, n_steps + 1)
    assert out.cumulative_loss.shape == (cfg.n_paths, n_steps + 1)
    assert out.default_times.shape == (cfg.n_paths, cfg.n_obligors)
    assert out.net_rent.shape == (cfg.n_paths, n_steps)


def test_engine_is_deterministic_under_same_seed(baseline_inputs: tuple) -> None:
    cfg, gbm, vas, lgd_p, tranches, coupons = baseline_inputs
    a = run_simulation(cfg, gbm=gbm, vasicek=vas, lgd=lgd_p, tranches=tranches, coupons=coupons)
    b = run_simulation(cfg, gbm=gbm, vasicek=vas, lgd=lgd_p, tranches=tranches, coupons=coupons)
    np.testing.assert_array_equal(a.price_paths, b.price_paths)
    np.testing.assert_array_equal(a.cumulative_loss, b.cumulative_loss)


def test_seeds_are_decorrelated() -> None:
    streams = seeds.make_streams(0)
    a = streams.price.standard_normal(10000)
    b = streams.credit.standard_normal(10000)
    corr = float(np.corrcoef(a, b)[0, 1])
    assert abs(corr) < 0.05


def test_cumulative_loss_is_non_decreasing(baseline_inputs: tuple) -> None:
    cfg, gbm, vas, lgd_p, tranches, coupons = baseline_inputs
    out = run_simulation(cfg, gbm=gbm, vasicek=vas, lgd=lgd_p, tranches=tranches, coupons=coupons)
    diffs = np.diff(out.cumulative_loss, axis=1)
    assert (diffs >= -1e-12).all()


def test_terminal_value_matches_surviving_fraction_times_market_price(
    baseline_inputs: tuple,
) -> None:
    cfg, gbm, vas, lgd_p, tranches, coupons = baseline_inputs
    out = run_simulation(cfg, gbm=gbm, vasicek=vas, lgd=lgd_p, tranches=tranches, coupons=coupons)
    expected = (1.0 - out.cumulative_loss[:, -1]) * out.price_paths[:, -1]
    np.testing.assert_allclose(out.terminal_value, expected)


def test_qmc_mode_recovers_same_marginal_pd(baseline_inputs: tuple) -> None:
    cfg, gbm, vas, lgd_p, tranches, coupons = baseline_inputs
    plain = run_simulation(cfg, gbm=gbm, vasicek=vas, lgd=lgd_p, tranches=tranches, coupons=coupons)
    cfg_qmc = replace(cfg, use_qmc=True)
    qmc_out = run_simulation(
        cfg_qmc, gbm=gbm, vasicek=vas, lgd=lgd_p, tranches=tranches, coupons=coupons
    )
    # Both should produce marginal PD close to pd_terminal.
    assert abs((plain.default_times <= cfg.horizon_years).mean() - cfg.pd_terminal) < 0.02
    assert abs((qmc_out.default_times <= cfg.horizon_years).mean() - cfg.pd_terminal) < 0.02


def test_qmc_helper_returns_correct_shape() -> None:
    sample = _gaussian_copula_qmc(
        SimulationConfig(
            n_obligors=30,
            horizon_years=10.0,
            steps_per_year=12,
            par=1.0,
            gross_yield=0.04,
            maintenance_pct=0.01,
            initial_rate=0.03,
            pd_terminal=0.10,
            credit_model="gaussian_copula",
            rho=0.10,
            n_paths=200,
            use_qmc=True,
        ),
        qmc_seed=0,
    )
    assert sample.shape == (200, 30)


def test_engine_supports_student_t(baseline_inputs: tuple) -> None:
    cfg, gbm, vas, lgd_p, tranches, coupons = baseline_inputs
    cfg_t = replace(cfg, credit_model="student_t_copula", nu=5.0)
    out = run_simulation(cfg_t, gbm=gbm, vasicek=vas, lgd=lgd_p, tranches=tranches, coupons=coupons)
    assert out.default_times.shape == (cfg.n_paths, cfg.n_obligors)


def test_engine_supports_cox_intensity(baseline_inputs: tuple) -> None:
    cfg, gbm, vas, lgd_p, tranches, coupons = baseline_inputs
    cfg_c = replace(cfg, credit_model="cox_intensity")
    out = run_simulation(cfg_c, gbm=gbm, vasicek=vas, lgd=lgd_p, tranches=tranches, coupons=coupons)
    finite = out.default_times[np.isfinite(out.default_times)]
    if finite.size > 0:
        assert (finite >= 0).all()
        assert (finite <= cfg.horizon_years).all()


def test_constant_discount_convention_uses_initial_rate(baseline_inputs: tuple) -> None:
    cfg, gbm, vas, lgd_p, tranches, coupons = baseline_inputs
    constant_cfg = replace(cfg, discount_convention="constant_at_initial_rate")
    out = run_simulation(
        constant_cfg, gbm=gbm, vasicek=vas, lgd=lgd_p, tranches=tranches, coupons=coupons
    )
    n_steps = out.n_steps
    dt = out.dt
    t_grid = np.arange(n_steps + 1) * dt
    expected = np.exp(-cfg.initial_rate * t_grid)
    # Every path has identical (deterministic) discount factors.
    np.testing.assert_allclose(
        out.discount_factors, np.broadcast_to(expected, out.discount_factors.shape)
    )


def test_vasicek_discount_convention_path_dependent(baseline_inputs: tuple) -> None:
    cfg, gbm, vas, lgd_p, tranches, coupons = baseline_inputs
    out = run_simulation(cfg, gbm=gbm, vasicek=vas, lgd=lgd_p, tranches=tranches, coupons=coupons)
    # By default the convention is "vasicek_simulated" → each path has its own
    # discount-factor trajectory, so paths should disagree past t=0.
    std_across_paths_at_T = float(out.discount_factors[:, -1].std(ddof=1))
    assert std_across_paths_at_T > 1e-6


def test_engine_rejects_invalid_inputs() -> None:
    bad_cfg = SimulationConfig(
        n_obligors=0,  # invalid
        horizon_years=10.0,
        steps_per_year=12,
        par=1.0,
        gross_yield=0.04,
        maintenance_pct=0.01,
        initial_rate=0.03,
        pd_terminal=0.10,
        credit_model="gaussian_copula",
        rho=0.10,
        n_paths=10,
    )
    with pytest.raises(ValueError):
        run_simulation(
            bad_cfg,
            gbm=GBMParams(mu=0.02, sigma=0.05),
            vasicek=VasicekParams(a=0.20, b=0.025, sigma_r=0.01),
            lgd=BetaLGDParams(mean=0.80, std=0.15),
            tranches=[
                Tranche("equity", 0.0, 0.25),
                Tranche("mezzanine", 0.25, 0.60),
                Tranche("senior", 0.60, 1.0),
            ],
            coupons={"senior": 0.03, "mezzanine": 0.06, "equity": 0.0},
        )
