"""Notional-depleting tranche waterfall (Andersen-Sidenius-Basu 2003).

Single entry point :func:`run` ingests a Monte Carlo loss path and produces
the tranche-level cash flows that the pricing layer discounts.
The mechanics are:

1. **Loss allocation** (stop-loss per tranche): at every period the cumulative
   portfolio loss ``L(t)`` is split across the stack via
   :func:`tranche_pricing.waterfall.tranches.loss_to_tranche`. Each tranche's
   remaining notional is ``thickness - loss_absorbed``.
2. **Interest waterfall**: the period's net rent ``net_rent_t`` is distributed
   sequentially — senior coupon (``c_S * N_S(t) * dt``) first, then mezzanine,
   then equity receives the residual (which can be negative when defaults
   exceed the available cushion).
3. **Principal waterfall**: at the horizon the building's terminal value
   ``terminal_value`` is distributed sequentially — senior gets back its
   surviving notional, then mezzanine, then equity collects whatever is
   left.

The implementation is fully vectorised over the Monte Carlo simulation axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .tranches import Tranche, loss_to_tranche


@dataclass(slots=True)
class WaterfallOutcome:
    """Per-tranche cash flow paths returned by :func:`run`.

    All arrays are indexed along axis 0 by Monte Carlo path.

    Attributes
    ----------
    interest_cash_flows
        ``name -> (n_paths, n_periods)`` coupon actually paid per period.
        For mezzanine / senior tranches this matches the promised coupon
        whenever ``net_rent`` is sufficient. The equity tranche absorbs the
        residual (positive or negative).
    principal_cash_flows
        ``name -> (n_paths,)`` payback at the horizon from the terminal sale.
    notional_path
        ``name -> (n_paths, n_periods+1)`` remaining notional at period END.
    loss_path
        ``name -> (n_paths, n_periods+1)`` cumulative loss absorbed by the
        tranche up to period END.
    """

    interest_cash_flows: dict[str, NDArray[np.float64]]
    principal_cash_flows: dict[str, NDArray[np.float64]]
    notional_path: dict[str, NDArray[np.float64]]
    loss_path: dict[str, NDArray[np.float64]]
    reserve_account: NDArray[np.float64] | None = None


def run(
    *,
    cumulative_loss: NDArray[np.float64],
    net_rent: NDArray[np.float64],
    terminal_value: NDArray[np.float64],
    tranches: list[Tranche],
    coupons: dict[str, float],
    par: float,
    dt: float,
    oc_test_enabled: bool = False,
    trigger_oc: float = 1.10,
    target_oc: float = 1.15,
) -> WaterfallOutcome:
    """Execute the full ASB (2003) tranche waterfall for an MC batch.

    Parameters
    ----------
    cumulative_loss
        Shape ``(n_paths, n_periods + 1)``. Loss fraction of par at every
        period END, including the initial zero at index 0.
    net_rent
        Shape ``(n_paths, n_periods)``. Net cash inflow for the period
        (gross rent net of maintenance, insurance premium and any other
        ancillary cost). Must be non-negative; losses are accounted for
        separately via ``cumulative_loss``.
    terminal_value
        Shape ``(n_paths,)``. Sale proceeds at the horizon (e.g. the
        simulated building value at ``T``).
    tranches
        Tranche stack ordered from junior to senior. Must tile ``[0, 1]``.
    coupons
        ``{tranche_name: coupon}`` annualised coupon paid pro-rata on the
        tranche notional. Equity is allowed to have a coupon of 0 (residual
        claim).
    par
        Total portfolio par in monetary units (e.g. the building value at
        ``t = 0``). All ``cumulative_loss`` and ``thickness`` values are
        fractions of this par; the cash flows are returned in the same units.
    dt
        Period length in years.

    Returns
    -------
    WaterfallOutcome
        Vectorised cash-flow paths per tranche.
    """
    if cumulative_loss.ndim != 2:
        raise ValueError("cumulative_loss must be 2D of shape (n_paths, n_periods+1).")
    if net_rent.ndim != 2:
        raise ValueError("net_rent must be 2D of shape (n_paths, n_periods).")
    if cumulative_loss.shape[1] != net_rent.shape[1] + 1:
        raise ValueError(
            "cumulative_loss must have one more time index than net_rent "
            "(period END including the t=0 zero)."
        )
    if not (cumulative_loss.shape[0] == net_rent.shape[0] == terminal_value.shape[0]):
        raise ValueError("All inputs must share the same n_paths along axis 0.")
    if par <= 0 or dt <= 0:
        raise ValueError("par and dt must both be positive.")

    n_paths, n_steps_plus_one = cumulative_loss.shape
    n_periods = n_steps_plus_one - 1
    sorted_tr = sorted(tranches, key=lambda t: t.attach)

    # ----- 1. Loss allocation per tranche along the time axis. ----------- #
    loss_path: dict[str, NDArray[np.float64]] = {}
    notional_path: dict[str, NDArray[np.float64]] = {}
    for tr in sorted_tr:
        loss = loss_to_tranche(cumulative_loss, tr)  # (n_paths, n_periods+1)
        loss_arr = np.asarray(loss, dtype=np.float64)
        loss_path[tr.name] = loss_arr
        notional_path[tr.name] = par * (tr.thickness - loss_arr)

    # ----- 2. Interest waterfall. ---------------------------------------- #
    interest_cash_flows: dict[str, NDArray[np.float64]] = {
        tr.name: np.zeros((n_paths, n_periods), dtype=np.float64) for tr in sorted_tr
    }

    # Junior to senior, but coupons are paid senior FIRST (priority). So we
    # iterate the stack from senior to junior when distributing the cash.
    seniority_order = list(reversed(sorted_tr))  # senior, mezz, equity
    senior_name = seniority_order[0].name
    # Reserve account (path-wise): build up cash trapped by the OC test, paid
    # out to equity once the OC ratio recovers. Always tracked but zero when
    # ``oc_test_enabled`` is False.
    reserve_account = np.zeros((n_paths, n_periods + 1), dtype=np.float64)

    for k in range(n_periods):
        remaining_cash = net_rent[:, k].astype(np.float64).copy()
        for tr in seniority_order:
            promised = coupons[tr.name] * notional_path[tr.name][:, k] * dt
            if tr is seniority_order[-1]:
                # Equity (most junior) absorbs the residual — but the OC test,
                # when active, may divert it into the reserve account first.
                equity_residual = remaining_cash
                if oc_test_enabled:
                    total_notional = sum(notional_path[t.name][:, k] for t in sorted_tr)
                    senior_notional = notional_path[senior_name][:, k]
                    safe_senior = np.where(senior_notional > 0, senior_notional, 1.0)
                    oc_ratio = total_notional / safe_senior
                    trap_mask = oc_ratio < trigger_oc
                    release_mask = (oc_ratio >= target_oc) & (reserve_account[:, k] > 0)
                    # 1) Trap equity cash flow when OC is too low.
                    trapped = np.where(trap_mask, np.maximum(equity_residual, 0.0), 0.0)
                    equity_after_trap = equity_residual - trapped
                    # 2) Release reserve when OC has recovered.
                    released = np.where(release_mask, reserve_account[:, k], 0.0)
                    reserve_account[:, k + 1] = reserve_account[:, k] + trapped - released
                    interest_cash_flows[tr.name][:, k] = equity_after_trap + released
                else:
                    reserve_account[:, k + 1] = reserve_account[:, k]
                    interest_cash_flows[tr.name][:, k] = equity_residual
            else:
                paid = np.minimum(remaining_cash, promised)
                interest_cash_flows[tr.name][:, k] = paid
                remaining_cash = remaining_cash - paid

    # ----- 3. Principal waterfall at terminal date. ---------------------- #
    principal_cash_flows: dict[str, NDArray[np.float64]] = {}
    remaining = terminal_value.astype(np.float64).copy()
    # Pay senior first, then mezz, equity last.
    for tr in seniority_order:
        notional_T = notional_path[tr.name][:, -1]
        if tr is seniority_order[-1]:
            # Equity gets whatever remains.
            principal_cash_flows[tr.name] = remaining
        else:
            paid = np.minimum(remaining, notional_T)
            principal_cash_flows[tr.name] = paid
            remaining = remaining - paid

    # Release any remaining reserve at terminal into equity principal.
    if oc_test_enabled:
        principal_cash_flows[seniority_order[-1].name] = (
            principal_cash_flows[seniority_order[-1].name] + reserve_account[:, -1]
        )

    return WaterfallOutcome(
        interest_cash_flows=interest_cash_flows,
        principal_cash_flows=principal_cash_flows,
        notional_path=notional_path,
        loss_path=loss_path,
        reserve_account=reserve_account,
    )


__all__ = ["WaterfallOutcome", "run"]
