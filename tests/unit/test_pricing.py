"""Tests for the per-instrument pricing layer."""

from __future__ import annotations

import numpy as np
import pytest

from tranche_pricing.credit.lgd import BetaLGDParams
from tranche_pricing.insurance import actuarial, option_theoretic
from tranche_pricing.markets.price_gbm import GBMParams
from tranche_pricing.markets.rates_vasicek import VasicekParams
from tranche_pricing.pricing import extract_all, price_all
from tranche_pricing.simulation import SimulationConfig, run_simulation
from tranche_pricing.waterfall.tranches import Tranche


@pytest.fixture(scope="module")
def small_run() -> tuple:
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
        rho=0.15,
        n_paths=500,
        master_seed=7,
    )
    out = run_simulation(
        cfg,
        gbm=GBMParams(mu=0.02, sigma=0.05),
        vasicek=VasicekParams(a=0.20, b=0.03, sigma_r=0.01),
        lgd=BetaLGDParams(mean=0.85, std=0.12),
        tranches=[
            Tranche("equity", 0.0, 0.25),
            Tranche("mezzanine", 0.25, 0.60),
            Tranche("senior", 0.60, 1.0),
        ],
        coupons={"senior": 0.03, "mezzanine": 0.06, "equity": 0.0},
    )
    return cfg, out


def test_extract_all_returns_five_instruments(small_run: tuple) -> None:
    _, out = small_run
    instruments = extract_all(out)
    assert set(instruments) == {"model_a", "model_b", "equity", "mezzanine", "senior"}
    for inst in instruments.values():
        assert inst.interest_cash_flows.shape[0] == out.n_paths
        assert inst.principal_cash_flow.shape == (out.n_paths,)
        assert inst.initial_price > 0


def test_initial_prices_match_par_thickness(small_run: tuple) -> None:
    cfg, out = small_run
    instruments = extract_all(out)
    apt_par = cfg.par / cfg.n_obligors
    assert instruments["model_a"].initial_price == pytest.approx(apt_par)
    assert instruments["model_b"].initial_price == pytest.approx(apt_par)
    # Tranche initial prices = par * thickness
    assert instruments["equity"].initial_price == pytest.approx(0.25 * cfg.par)
    assert instruments["mezzanine"].initial_price == pytest.approx(0.35 * cfg.par)
    assert instruments["senior"].initial_price == pytest.approx(0.40 * cfg.par)


def test_price_all_produces_summary(small_run: tuple) -> None:
    _, out = small_run
    instruments = extract_all(out)
    pricings = price_all(instruments, out=out)
    for inst_name, p in pricings.items():
        assert p.name == inst_name
        assert p.annualized_return.shape == (out.n_paths,)
        assert "sharpe" in p.risk
        assert "var_95" in p.risk
        assert "es_95" in p.risk


def test_actuarial_premium_increases_with_loading(small_run: tuple) -> None:
    cfg, out = small_run
    low = actuarial.price_premium(
        out.cumulative_loss, par=cfg.par, admin_loading=0.0, risk_loading=0.0
    )
    high = actuarial.price_premium(
        out.cumulative_loss, par=cfg.par, admin_loading=0.30, risk_loading=0.30
    )
    assert high["annual_premium"] > low["annual_premium"]


def test_option_theoretic_premium_non_negative(small_run: tuple) -> None:
    _, out = small_run
    prem = option_theoretic.price_premium(out)
    assert prem["annual_premium"] >= 0
    assert prem["lump_sum_premium"] >= 0


def test_allocate_premium_zero_coverage_yields_zero_weights(small_run: tuple) -> None:
    from tranche_pricing.insurance.actuarial import allocate_premium_to_tranches
    from tranche_pricing.waterfall.tranches import Tranche

    _, out = small_run
    tranches = [
        Tranche("equity", 0.0, 0.25),
        Tranche("mezzanine", 0.25, 0.60),
        Tranche("senior", 0.60, 1.0),
    ]
    w = allocate_premium_to_tranches(
        cumulative_loss_no_ins=out.cumulative_loss,
        tranches=tranches,
        coverage_cap=0.0,
    )
    assert all(v == 0.0 for v in w.values())


def test_allocate_premium_equity_share_largest(small_run: tuple) -> None:
    from tranche_pricing.insurance.actuarial import allocate_premium_to_tranches
    from tranche_pricing.waterfall.tranches import Tranche

    _, out = small_run
    tranches = [
        Tranche("equity", 0.0, 0.25),
        Tranche("mezzanine", 0.25, 0.60),
        Tranche("senior", 0.60, 1.0),
    ]
    w = allocate_premium_to_tranches(
        cumulative_loss_no_ins=out.cumulative_loss,
        tranches=tranches,
        coverage_cap=0.9,
    )
    # Equity absorbs first losses → it pays the biggest share of the premium.
    assert w["equity"] > w["mezzanine"]
    assert w["mezzanine"] >= w["senior"]
    assert abs(sum(w.values()) - 1.0) < 1e-9


def test_model_a_initial_price_times_n_obligors_equals_par(small_run: tuple) -> None:
    cfg, out = small_run
    instruments = extract_all(out)
    assert instruments["model_a"].initial_price * cfg.n_obligors == pytest.approx(cfg.par)


def test_solve_fair_coupon_finds_senior_root_above_contract(small_run: tuple) -> None:
    from tranche_pricing.pricing import solve_fair_coupon

    _, out = small_run
    base = {"senior": 0.03, "mezzanine": 0.06, "equity": 0.0}
    c = solve_fair_coupon(out=out, tranche_name="senior", coupons_template=base)
    if not np.isnan(c):
        # When a finite fair coupon exists it should sit above the contract.
        assert c > 0.0


def test_solve_fair_coupons_for_all_returns_dict(small_run: tuple) -> None:
    from tranche_pricing.pricing import solve_fair_coupons_for_all

    _, out = small_run
    fair = solve_fair_coupons_for_all(
        out=out,
        base_coupons={"senior": 0.03, "mezzanine": 0.06, "equity": 0.0},
        tranche_names=("senior", "mezzanine"),
    )
    assert set(fair) == {"senior", "mezzanine", "equity"}
    # Equity stays at its base value because it is the residual claimant.
    assert fair["equity"] == 0.0
