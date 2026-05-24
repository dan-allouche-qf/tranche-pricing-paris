"""Shared interface for the three credit-risk models.

Every credit model exposes a ``simulate_default_times`` function with the
signature defined below; the simulation engine and the pricing layer treat
the three models interchangeably so that the comparison study in the
working paper is genuinely apples-to-apples.

Default times are expressed in years from contract inception and ``np.inf`` is
used for obligors that never default within the modelled horizon. The
convention follows Andersen-Sidenius-Basu (2003).
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class CreditModel(Protocol):
    """Structural protocol every credit model implements."""

    def simulate_default_times(
        self,
        *,
        horizon_years: float,
        pd_terminal: float,
        n_sims: int,
        n_obligors: int,
        rng: np.random.Generator,
    ) -> NDArray[np.float64]: ...


def cumulative_default_indicator(default_times: NDArray[np.float64], t: float) -> NDArray[np.bool_]:
    """Return a boolean mask for ``default_times <= t`` (vectorised over an array)."""
    return default_times <= t


def constant_hazard(pd_terminal: float, horizon_years: float) -> float:
    """Solve for the constant hazard ``h`` matching ``pd_terminal`` over ``horizon_years``.

    Under exponential survival the cumulative PD is ``1 - exp(-h t)``. Inverting
    gives ``h = -log(1 - pd_terminal) / horizon_years``.
    """
    if not 0 < pd_terminal < 1:
        raise ValueError("pd_terminal must be in (0, 1).")
    if horizon_years <= 0:
        raise ValueError("horizon_years must be positive.")
    return -float(np.log(1.0 - pd_terminal)) / float(horizon_years)


def default_time_from_uniform(
    u: NDArray[np.float64], horizon_years: float, pd_terminal: float
) -> NDArray[np.float64]:
    """Convert a uniform marginal ``u`` to a default time under constant hazard.

    Solves ``u = 1 - exp(-h tau)`` for ``tau``, capping at ``inf`` when the
    obligor's u-score exceeds the terminal cumulative PD.
    """
    h = constant_hazard(pd_terminal, horizon_years)
    safe = np.clip(u, a_min=0.0, a_max=1.0 - 1e-12)
    tau = -np.log(1.0 - safe) / h
    tau = np.where(safe < pd_terminal, tau, np.inf)
    return tau.astype(np.float64)


__all__ = [
    "CreditModel",
    "constant_hazard",
    "cumulative_default_indicator",
    "default_time_from_uniform",
]
