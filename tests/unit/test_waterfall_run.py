"""End-to-end tests for the ASB tranche waterfall."""

from __future__ import annotations

import numpy as np
import pytest

from tranche_pricing.waterfall import andersen_sidenius, loss_paths
from tranche_pricing.waterfall.tranches import Tranche


def _standard_stack() -> list[Tranche]:
    return [
        Tranche("equity", 0.0, 0.25),
        Tranche("mezzanine", 0.25, 0.60),
        Tranche("senior", 0.60, 1.00),
    ]


def test_run_zero_default_senior_fully_repaid_first() -> None:
    """With zero defaults, the principal waterfall fully repays senior at par."""
    n_paths, n_periods = 5, 10
    cum_loss = np.zeros((n_paths, n_periods + 1))
    net_rent = np.full((n_paths, n_periods), 50_000.0)
    terminal = np.full(n_paths, 1_000_000.0)
    stack = _standard_stack()
    coupons = {"senior": 0.03, "mezzanine": 0.06, "equity": 0.0}

    out = andersen_sidenius.run(
        cumulative_loss=cum_loss,
        net_rent=net_rent,
        terminal_value=terminal,
        tranches=stack,
        coupons=coupons,
        par=1_000_000.0,
        dt=1.0,
    )
    # Senior has 0.40 thickness × par = 400_000 par, fully repaid.
    np.testing.assert_allclose(out.principal_cash_flows["senior"], 400_000.0)
    np.testing.assert_allclose(out.principal_cash_flows["mezzanine"], 350_000.0)
    # Equity gets what's left = 1_000_000 - 400_000 - 350_000 = 250_000.
    np.testing.assert_allclose(out.principal_cash_flows["equity"], 250_000.0)


def test_run_equity_wiped_when_loss_exceeds_25pct() -> None:
    n_paths, n_periods = 1, 5
    cum_loss = np.zeros((n_paths, n_periods + 1))
    cum_loss[0, 1:] = np.linspace(0.10, 0.30, n_periods)  # crosses 0.25 mid-horizon
    net_rent = np.zeros((n_paths, n_periods))
    terminal = np.array([700_000.0])  # 30% loss → 70% terminal value of par 1M
    stack = _standard_stack()
    coupons = {"senior": 0.0, "mezzanine": 0.0, "equity": 0.0}

    out = andersen_sidenius.run(
        cumulative_loss=cum_loss,
        net_rent=net_rent,
        terminal_value=terminal,
        tranches=stack,
        coupons=coupons,
        par=1_000_000.0,
        dt=1.0,
    )
    # Equity fully wiped (loss 0.25 of 0.25 thickness)
    assert out.notional_path["equity"][0, -1] == pytest.approx(0.0)
    # Mezz hit by 0.30 - 0.25 = 0.05, remaining = 0.35 - 0.05 = 0.30 of par = 300_000
    assert out.notional_path["mezzanine"][0, -1] == pytest.approx(300_000.0)
    # Senior untouched
    assert out.notional_path["senior"][0, -1] == pytest.approx(400_000.0)


def test_loss_invariant_sum_tranche_losses_equals_aggregate() -> None:
    rng = np.random.default_rng(0)
    n_paths, n_periods = 100, 10
    cum_loss = np.zeros((n_paths, n_periods + 1))
    cum_loss[:, 1:] = np.sort(rng.uniform(0, 0.7, size=(n_paths, n_periods)), axis=1)
    net_rent = np.zeros((n_paths, n_periods))
    terminal = np.zeros(n_paths)
    stack = _standard_stack()
    coupons = {"senior": 0.0, "mezzanine": 0.0, "equity": 0.0}

    out = andersen_sidenius.run(
        cumulative_loss=cum_loss,
        net_rent=net_rent,
        terminal_value=terminal,
        tranches=stack,
        coupons=coupons,
        par=1.0,
        dt=1.0,
    )
    total_tranche_loss = sum(out.loss_path[name][:, -1] for name in out.loss_path)
    aggregate_loss = cum_loss[:, -1]
    np.testing.assert_allclose(total_tranche_loss, aggregate_loss, atol=1e-12)


def test_interest_waterfall_pays_senior_first_then_mezz() -> None:
    """When net rent equals senior + half of mezz coupon, mezz gets only half paid."""
    n_paths, n_periods = 1, 1
    cum_loss = np.zeros((n_paths, n_periods + 1))
    net_rent = np.array([[400_000 * 0.03 + 350_000 * 0.04]])  # senior + half mezz
    terminal = np.array([1_000_000.0])
    stack = _standard_stack()
    coupons = {"senior": 0.03, "mezzanine": 0.08, "equity": 0.0}  # mezz promised 0.08

    out = andersen_sidenius.run(
        cumulative_loss=cum_loss,
        net_rent=net_rent,
        terminal_value=terminal,
        tranches=stack,
        coupons=coupons,
        par=1_000_000.0,
        dt=1.0,
    )
    assert out.interest_cash_flows["senior"][0, 0] == pytest.approx(0.03 * 400_000)
    # Mezz was promised 0.08 * 350_000 = 28_000 but only 0.04 * 350_000 = 14_000 in the pot
    assert out.interest_cash_flows["mezzanine"][0, 0] == pytest.approx(0.04 * 350_000)
    # Equity gets the residual = 0
    assert out.interest_cash_flows["equity"][0, 0] == pytest.approx(0.0)


def test_input_validation() -> None:
    with pytest.raises(ValueError):
        andersen_sidenius.run(
            cumulative_loss=np.zeros(4),  # 1D, wrong
            net_rent=np.zeros((1, 3)),
            terminal_value=np.zeros(1),
            tranches=_standard_stack(),
            coupons={"senior": 0.0, "mezzanine": 0.0, "equity": 0.0},
            par=1.0,
            dt=1.0,
        )


def test_input_validation_catches_path_count_mismatch_with_terminal() -> None:
    """A shape mismatch on the ``terminal_value`` axis must raise.

    The validation uses a grouped equality predicate so that an axis-0
    disagreement on any one of the three inputs is detected.
    """
    n_periods = 3
    with pytest.raises(ValueError):
        andersen_sidenius.run(
            # cumulative_loss and net_rent both have 2 paths along axis 0 …
            cumulative_loss=np.zeros((2, n_periods + 1)),
            net_rent=np.zeros((2, n_periods)),
            # … but terminal_value disagrees on axis 0.
            terminal_value=np.zeros(3),
            tranches=_standard_stack(),
            coupons={"senior": 0.0, "mezzanine": 0.0, "equity": 0.0},
            par=1.0,
            dt=1.0,
        )


def test_cumulative_loss_path_from_defaults() -> None:
    """A single default at year 3 with LGD 0.85 should make L(t) jump in year 3."""
    default_times = np.array([[3.5, np.inf, np.inf, np.inf]])  # 1 path, 4 obligors
    lgd = np.array([[0.85, 0.0, 0.0, 0.0]])
    path = loss_paths.cumulative_loss_path(
        default_times=default_times,
        lgd_samples=lgd,
        n_periods=10,
        dt=1.0,
    )
    # Single obligor of 4 defaults: contribution = 0.85 / 4 = 0.2125 at period >= 4
    assert path.shape == (1, 11)
    assert path[0, 0] == 0.0
    assert path[0, 3] == 0.0  # default at t=3.5 falls in period 3 (END idx 4)
    assert path[0, 4] == pytest.approx(0.85 / 4)
    assert path[0, -1] == pytest.approx(0.85 / 4)


def test_oc_test_diverts_equity_when_oc_ratio_drops() -> None:
    """When the OC ratio is below trigger, equity cash is trapped."""
    n_paths, n_periods = 1, 4
    # Force cumulative loss high enough that the OC ratio < trigger.
    cum_loss = np.zeros((n_paths, n_periods + 1))
    cum_loss[0, 1:] = np.linspace(0.20, 0.30, n_periods)
    net_rent = np.full((n_paths, n_periods), 50_000.0)
    terminal = np.array([1_000_000.0])
    stack = _standard_stack()
    coupons = {"senior": 0.05, "mezzanine": 0.05, "equity": 0.0}

    no_oc = andersen_sidenius.run(
        cumulative_loss=cum_loss,
        net_rent=net_rent,
        terminal_value=terminal,
        tranches=stack,
        coupons=coupons,
        par=1_000_000.0,
        dt=1.0,
    )
    with_oc = andersen_sidenius.run(
        cumulative_loss=cum_loss,
        net_rent=net_rent,
        terminal_value=terminal,
        tranches=stack,
        coupons=coupons,
        par=1_000_000.0,
        dt=1.0,
        oc_test_enabled=True,
        trigger_oc=1.20,
        target_oc=1.25,
    )
    # With OC active the equity interest cash flow should be lower (some
    # residual was trapped).
    assert with_oc.interest_cash_flows["equity"].sum() <= no_oc.interest_cash_flows["equity"].sum()
    assert with_oc.reserve_account is not None
    # But the equity principal should compensate (trapped reserve released at terminal).
    assert (
        with_oc.principal_cash_flows["equity"].sum() >= no_oc.principal_cash_flows["equity"].sum()
    )


def test_cumulative_loss_path_aggregates_multiple_defaults() -> None:
    default_times = np.array([[1.5, 2.5, 8.0]])  # 1 path, 3 obligors
    lgd = np.array([[0.50, 0.80, 0.90]])
    path = loss_paths.cumulative_loss_path(
        default_times=default_times,
        lgd_samples=lgd,
        n_periods=10,
        dt=1.0,
    )
    # End of period 1 (idx 2): 0.50/3
    assert path[0, 2] == pytest.approx(0.50 / 3)
    # End of period 2 (idx 3): + 0.80/3
    assert path[0, 3] == pytest.approx((0.50 + 0.80) / 3)
    # End of period 8 (idx 9): + 0.90/3
    assert path[0, 9] == pytest.approx((0.50 + 0.80 + 0.90) / 3)
