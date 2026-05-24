"""Tranche definitions and loss-allocation primitives.

A tranche is identified by its (attach, detach) points on the [0, 1]
cumulative-loss axis. Given a cumulative portfolio loss ``L`` (also expressed
as a fraction of par), the tranche absorbs::

    loss(L) = min(max(L - attach, 0), detach - attach).

This is the "stop-loss" / vertical-spread payoff that underlies every CDO
waterfall in the literature (Andersen-Sidenius-Basu 2003, Brigo-Pallavicini,
etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class Tranche:
    """A tranche described by its attach / detach points and label."""

    name: str
    attach: float
    detach: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.attach <= 1.0:
            raise ValueError(f"Tranche {self.name!r}: attach must be in [0, 1].")
        if not 0.0 <= self.detach <= 1.0:
            raise ValueError(f"Tranche {self.name!r}: detach must be in [0, 1].")
        if self.attach >= self.detach:
            raise ValueError(
                f"Tranche {self.name!r}: attach={self.attach} must be < detach={self.detach}."
            )

    @property
    def thickness(self) -> float:
        """Notional thickness of the tranche, expressed as a fraction of par."""
        return self.detach - self.attach


def loss_to_tranche(
    cumulative_loss: float | NDArray[np.float64],
    tranche: Tranche,
) -> float | NDArray[np.float64]:
    """Return the loss absorbed by ``tranche`` given the aggregate ``cumulative_loss``.

    Both inputs and outputs are fractions of par. Works elementwise on
    array-valued ``cumulative_loss``.
    """
    excess = np.maximum(np.asarray(cumulative_loss, dtype=float) - tranche.attach, 0.0)
    capped = np.minimum(excess, tranche.thickness)
    if isinstance(cumulative_loss, np.ndarray):
        return capped
    return float(capped)


def allocate_loss_across_stack(
    cumulative_loss: float | NDArray[np.float64],
    tranches: list[Tranche],
) -> dict[str, float | NDArray[np.float64]]:
    """Allocate the aggregate loss across an ordered tranche stack.

    The stack is expected to tile ``[0, 1]`` (i.e. ``detach_i = attach_{i+1}``);
    when this is the case the sum of per-tranche losses equals
    ``cumulative_loss`` exactly. The function does not assume the stack is
    sorted; it sorts by ``attach`` internally.
    """
    sorted_tr = sorted(tranches, key=lambda t: t.attach)
    return {t.name: loss_to_tranche(cumulative_loss, t) for t in sorted_tr}


def tranche_pv01(
    *,
    notionals_over_time: NDArray[np.float64],
    coupon: float,
    discount_factors: NDArray[np.float64],
    dt: float,
) -> float:
    """Present value of a 1-bp coupon stream on a time-varying tranche notional.

    Parameters
    ----------
    notionals_over_time
        Notional at the END of each period, shape (n_periods,).
    coupon
        Annualised coupon rate paid pro-rata on the notional (in decimal).
    discount_factors
        Discount factor at each payment date, same length as ``notionals_over_time``.
    dt
        Period length in years.
    """
    if notionals_over_time.shape != discount_factors.shape:
        raise ValueError("notionals_over_time and discount_factors must have the same shape.")
    return float((coupon * dt * notionals_over_time * discount_factors).sum())


__all__ = [
    "Tranche",
    "allocate_loss_across_stack",
    "loss_to_tranche",
    "tranche_pv01",
]
