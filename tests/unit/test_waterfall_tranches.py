"""Tests for the tranche primitives and stop-loss allocation."""

from __future__ import annotations

import numpy as np
import pytest

from tranche_pricing.waterfall.tranches import (
    Tranche,
    allocate_loss_across_stack,
    loss_to_tranche,
)


def _standard_stack() -> list[Tranche]:
    return [
        Tranche("equity", 0.0, 0.25),
        Tranche("mezzanine", 0.25, 0.60),
        Tranche("senior", 0.60, 1.00),
    ]


def test_tranche_rejects_attach_outside_unit_interval() -> None:
    with pytest.raises(ValueError):
        Tranche("bad", -0.1, 0.5)
    with pytest.raises(ValueError):
        Tranche("bad", 0.0, 1.5)


def test_tranche_attach_must_be_below_detach() -> None:
    with pytest.raises(ValueError, match="must be <"):
        Tranche("bad", 0.5, 0.5)


def test_thickness_equals_detach_minus_attach() -> None:
    t = Tranche("mezz", 0.25, 0.60)
    assert t.thickness == pytest.approx(0.35)


# --------------------------------------------------------------------------- #
# Stop-loss payoff hand-computed cases                                        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "L, expected",
    [
        (0.00, 0.00),
        (0.10, 0.10),  # within equity
        (0.25, 0.25),  # equity exhausted exactly
        (0.40, 0.25),
        (1.00, 0.25),
    ],
)
def test_equity_stop_loss(L: float, expected: float) -> None:
    eq = Tranche("equity", 0.0, 0.25)
    assert loss_to_tranche(L, eq) == pytest.approx(expected)


@pytest.mark.parametrize(
    "L, expected",
    [
        (0.00, 0.00),
        (0.20, 0.00),  # equity still absorbing
        (0.40, 0.15),  # 0.15 spilled into mezz
        (0.60, 0.35),  # mezz exhausted
        (1.00, 0.35),
    ],
)
def test_mezzanine_stop_loss(L: float, expected: float) -> None:
    me = Tranche("mezz", 0.25, 0.60)
    assert loss_to_tranche(L, me) == pytest.approx(expected)


@pytest.mark.parametrize(
    "L, expected",
    [
        (0.00, 0.00),
        (0.50, 0.00),  # equity + mezz absorbing
        (0.70, 0.10),
        (1.00, 0.40),
    ],
)
def test_senior_stop_loss(L: float, expected: float) -> None:
    sr = Tranche("senior", 0.60, 1.00)
    assert loss_to_tranche(L, sr) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# Stack tiling invariant: sum of allocations = aggregate loss                 #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("L", [0.0, 0.1, 0.25, 0.45, 0.6, 0.85, 1.0])
def test_stack_allocations_sum_to_aggregate(L: float) -> None:
    stack = _standard_stack()
    alloc = allocate_loss_across_stack(L, stack)
    total = sum(alloc.values())
    assert total == pytest.approx(L)


def test_stack_allocations_vectorised() -> None:
    stack = _standard_stack()
    L = np.array([0.0, 0.1, 0.25, 0.45, 0.6, 1.0])
    alloc = allocate_loss_across_stack(L, stack)
    total = sum(alloc.values())
    np.testing.assert_allclose(total, L)
