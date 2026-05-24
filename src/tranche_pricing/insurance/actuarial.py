"""Actuarial pricing of the rental-default insurance.

The standard deviation principle (Bühlmann 1970, Wang 1996) prices the
insurance premium as

    P_actuarial = (1 + theta) * E[L_covered] + lambda * sigma(L_covered),

where ``L_covered`` is the loss actually borne by the insurer, ``theta`` is
an administrative loading (default 10 %) and ``lambda`` is a market risk
loading (default 15 %). For our setup ``L_covered`` per path is
``coverage_cap * cumulative_loss(T) * par`` — the fraction ``coverage_cap``
of all default losses is reimbursed by the insurer, the remainder is the
deductible.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def expected_loss(
    cumulative_loss: NDArray[np.float64],
    *,
    par: float,
    coverage_cap: float,
) -> NDArray[np.float64]:
    """Per-path covered loss in monetary units."""
    return coverage_cap * cumulative_loss[:, -1] * par


def price_premium(
    cumulative_loss: NDArray[np.float64],
    *,
    par: float,
    coverage_cap: float = 0.9,
    admin_loading: float = 0.10,
    risk_loading: float = 0.15,
    horizon_years: float = 10.0,
) -> dict[str, float]:
    """Return both the lump-sum and the annual-equivalent premium.

    The lump-sum premium covers losses over the full ``horizon_years``; the
    annual-equivalent divides by the horizon for direct comparison with
    market-quoted GLI premiums (~3 % of gross rent).
    """
    losses = expected_loss(cumulative_loss, par=par, coverage_cap=coverage_cap)
    mean = float(losses.mean())
    # std with ddof=1 is undefined for fewer than two samples; the SD principle
    # collapses to a pure expectation in that degenerate case.
    sd = float(losses.std(ddof=1)) if losses.size >= 2 else 0.0
    lump = (1.0 + admin_loading) * mean + risk_loading * sd
    annual = lump / horizon_years
    return {
        "expected_loss_per_path_mean": mean,
        "expected_loss_per_path_std": sd,
        "lump_sum_premium": float(lump),
        "annual_premium": float(annual),
        "admin_loading": float(admin_loading),
        "risk_loading": float(risk_loading),
        "coverage_cap": float(coverage_cap),
    }


def allocate_premium_to_tranches(
    *,
    cumulative_loss_no_ins: NDArray[np.float64],
    tranches: list,  # list[Tranche]; forward-declared to avoid a cycle
    coverage_cap: float,
) -> dict[str, float]:
    """Allocation weights of the insurance premium across tranches.

    Each tranche pays the premium proportional to the expected covered
    loss it would absorb under the no-insurance baseline. With the
    stop-loss payoff ``L_i(L) = min(max(L - a_i, 0), d_i - a_i)``,
    weight ``w_i = E[L_i(L(T))] * coverage_cap / sum_j ...``. Returns a
    dict keyed by tranche name, with weights summing to one (or to zero
    when ``coverage_cap == 0``).
    """
    from ..waterfall.tranches import loss_to_tranche

    if coverage_cap <= 0:
        return {t.name: 0.0 for t in tranches}
    terminal_loss = np.asarray(cumulative_loss_no_ins[:, -1], dtype=float)
    raw = {
        t.name: float(np.asarray(loss_to_tranche(terminal_loss, t)).mean() * coverage_cap)
        for t in tranches
    }
    total = sum(raw.values())
    if total <= 0:
        return dict.fromkeys(raw, 0.0)
    return {name: v / total for name, v in raw.items()}


__all__ = ["allocate_premium_to_tranches", "expected_loss", "price_premium"]
