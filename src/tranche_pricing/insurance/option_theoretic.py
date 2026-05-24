"""Option-theoretic pricing of rental-default insurance.

Insurance is modelled as a put on rental income: each period the insurer
pays ``max(strike_t - rent_t, 0)`` where ``strike_t`` is the promised rent
net of a deductible (typically one month). We discount these payoffs under
the simulated short-rate stream (treated as the numeraire) and average over
the Monte Carlo paths, giving the lump-sum premium. Setting

    PV(level_premium * coverage_cap) = PV(payoffs)

then determines the equivalent level annual premium so it can be compared
to the actuarial figure.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ..simulation.engine import SimulationOutput


def shortfall_payoffs(
    out: SimulationOutput,
    *,
    deductible_months: int = 1,
    coverage_cap: float = 0.9,
) -> NDArray[np.float64]:
    """Per-(path, period) insurance payoff.

    Strike = full unencumbered rent net of a per-period deductible. The
    franchise is ``deductible_months`` worth of monthly rent in total,
    amortised evenly across all periods so its present value is invariant
    to the simulation grid (monthly vs quarterly). Payoff per period is
    ``coverage_cap * max(strike - actual_rent, 0)``.
    """
    cfg = out.config
    dt = float(out.dt)
    n_steps = out.n_steps
    period_months = max(round(12 * dt), 1)

    full_rent = cfg.gross_yield * out.price_paths[:, :-1] * dt
    monthly_rent = full_rent / period_months
    deductible_per_period = (deductible_months * monthly_rent) / max(n_steps, 1)
    strike = np.maximum(full_rent - deductible_per_period, 0.0)
    shortfall = np.maximum(strike - out.net_rent, 0.0)
    return coverage_cap * shortfall


def price_premium(
    out: SimulationOutput,
    *,
    deductible_months: int = 1,
    coverage_cap: float = 0.9,
) -> dict[str, float]:
    """Risk-neutral-style premium pricing under the simulated short-rate measure.

    Returns the lump-sum premium PV of insurance payoffs, an annual-level
    premium and a per-rent ratio for comparison with quoted GLI rates.
    """
    cfg = out.config
    df = out.discount_factors[:, 1:]
    payoffs = shortfall_payoffs(out, deductible_months=deductible_months, coverage_cap=coverage_cap)
    pv_payoffs = (payoffs * df).sum(axis=1)
    lump = float(pv_payoffs.mean())

    full_rent = cfg.gross_yield * out.price_paths[:, :-1] * out.dt
    pv_full_rent = (full_rent * df).sum(axis=1).mean()
    annual = lump / cfg.horizon_years
    return {
        "lump_sum_premium": float(lump),
        "annual_premium": float(annual),
        "premium_pv_per_rent_pv": float(lump / pv_full_rent) if pv_full_rent > 0 else float("nan"),
        "coverage_cap": float(coverage_cap),
        "deductible_months": int(deductible_months),
    }


__all__ = ["price_premium", "shortfall_payoffs"]
