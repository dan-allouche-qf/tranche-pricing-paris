"""Per-instrument cash-flow extraction from a :class:`SimulationOutput`.

Five instruments are compared in the working paper:

* **Model A — Classical**: the investor buys one apartment of the building
  (1/n_obligors of par). Receives the corresponding share of net rent each
  period only while the apartment's tenant has not defaulted; at terminal
  the apartment is sold at its market value.
* **Model B — SAS pool**: pro-rata share of the aggregate SAS cash flows;
  full diversification across all apartments.
* **Equity tranche**: junior CDO tranche, absorbs first losses; receives
  residual interest after senior + mezzanine coupons.
* **Mezzanine tranche**: middle CDO tranche.
* **Senior tranche**: most protected CDO tranche.

For each instrument we expose ``InstrumentCashFlows`` containing the
per-path interest cash flow path, terminal payment and an ``initial_price``
benchmark used to compute returns / IRRs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from ..simulation.engine import SimulationOutput


@dataclass(slots=True)
class InstrumentCashFlows:
    """Periodic and terminal cash flows of one instrument, per MC path."""

    name: str
    initial_price: float
    interest_cash_flows: NDArray[np.float64]  # (n_paths, n_periods)
    principal_cash_flow: NDArray[np.float64]  # (n_paths,)

    @property
    def n_paths(self) -> int:
        return int(self.interest_cash_flows.shape[0])

    @property
    def n_periods(self) -> int:
        return int(self.interest_cash_flows.shape[1])


def extract_all(out: SimulationOutput) -> dict[str, InstrumentCashFlows]:
    """Build the cash-flow table for every comparison instrument.

    Each instrument's ``initial_price`` matches the "natural" capital
    deployed: the per-tranche par for tranches, ``par / n_obligors`` for the
    single-apartment Model A and Model B SAS share.
    """
    cfg = out.config
    n_obligors = cfg.n_obligors
    n_periods = out.net_rent.shape[1]
    n_paths = out.n_paths
    apt_par = cfg.par / n_obligors

    # ----- Model A — Classical: a single apartment, full default exposure -- #
    # We pick obligor #0 by symmetry across the simulation.
    dt = float(out.dt)
    alive_at_period_start = out.default_times[:, [0]] > np.arange(n_periods)[None, :] * dt
    # net rent per apartment (per period) = aggregate net rent / n_obligors
    per_apt_rent = out.net_rent / n_obligors
    model_a_int = per_apt_rent * alive_at_period_start
    # terminal value: receive apartment market value regardless of default
    model_a_term = out.price_paths[:, -1] / n_obligors

    # ----- Model B — SAS pool: pro-rata share of aggregate cash flows ------ #
    # Total aggregate interest paid to investors = sum of waterfall interest
    # cash flows across all tranches (which equals net_rent by construction).
    aggregate_interest = out.net_rent
    model_b_int = aggregate_interest / n_obligors
    model_b_term = out.terminal_value / n_obligors

    instruments: dict[str, InstrumentCashFlows] = {
        "model_a": InstrumentCashFlows(
            name="model_a",
            initial_price=apt_par,
            interest_cash_flows=model_a_int,
            principal_cash_flow=model_a_term,
        ),
        "model_b": InstrumentCashFlows(
            name="model_b",
            initial_price=apt_par,
            interest_cash_flows=model_b_int,
            principal_cash_flow=model_b_term,
        ),
    }

    # ----- Three tranches from the waterfall ------------------------------- #
    for tr_name, interest in out.waterfall.interest_cash_flows.items():
        principal = out.waterfall.principal_cash_flows[tr_name]
        thickness = out.waterfall.notional_path[tr_name][0, 0] / cfg.par
        instruments[tr_name] = InstrumentCashFlows(
            name=tr_name,
            initial_price=cfg.par * thickness,
            interest_cash_flows=interest,
            principal_cash_flow=principal,
        )

    # Ensure each instrument is shaped correctly.
    for inst in instruments.values():
        assert inst.interest_cash_flows.shape == (n_paths, n_periods)
        assert inst.principal_cash_flow.shape == (n_paths,)

    return instruments


__all__ = ["InstrumentCashFlows", "extract_all"]
