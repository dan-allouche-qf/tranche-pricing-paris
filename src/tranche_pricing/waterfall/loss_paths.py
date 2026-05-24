"""Build a cumulative-loss path from default times and per-obligor LGDs.

Given the credit layer's output ``default_times`` (shape ``(n_paths, n_obligors)``)
and a parallel sample of LGD fractions, we construct the cumulative
portfolio-loss path required by the tranche waterfall.

The total portfolio par is normalised to 1, so each obligor contributes
``1 / n_obligors`` of par and the loss recorded when obligor ``i`` defaults
is ``lgd_i / n_obligors``.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def cumulative_loss_path(
    *,
    default_times: NDArray[np.float64],
    lgd_samples: NDArray[np.float64],
    n_periods: int,
    dt: float,
) -> NDArray[np.float64]:
    """Construct the cumulative loss path over ``n_periods + 1`` time points.

    Parameters
    ----------
    default_times
        Shape ``(n_paths, n_obligors)``. ``np.inf`` for obligors that never
        default within the horizon.
    lgd_samples
        Shape ``(n_paths, n_obligors)`` LGD fractions in ``[0, 1]``.
    n_periods
        Number of periods in the simulation horizon.
    dt
        Period length in years.

    Returns
    -------
    NDArray[np.float64]
        Shape ``(n_paths, n_periods + 1)``. Cumulative portfolio-loss
        fraction at the END of each period (index 0 is the t=0 zero state).
    """
    if default_times.shape != lgd_samples.shape:
        raise ValueError("default_times and lgd_samples must have the same shape.")
    if n_periods <= 0 or dt <= 0:
        raise ValueError("n_periods and dt must be positive.")

    n_paths, n_obligors = default_times.shape
    finite_mask = np.isfinite(default_times)
    # Use a safe denominator to avoid casting inf/dt -> int (RuntimeWarning).
    safe_times = np.where(finite_mask, default_times, 0.0)
    period_index = np.where(
        finite_mask,
        np.floor(safe_times / dt).astype(int),
        n_periods,  # sentinel for never-default
    )
    period_index = np.where(period_index >= n_periods, n_periods, period_index)

    # contribution to period k = sum over obligors defaulting in period k
    per_obligor_loss = lgd_samples / n_obligors
    loss_per_period = np.zeros((n_paths, n_periods + 1), dtype=np.float64)
    for k in range(n_periods):
        mask = period_index == k
        loss_per_period[:, k + 1] = (per_obligor_loss * mask).sum(axis=1)

    return np.cumsum(loss_per_period, axis=1)


__all__ = ["cumulative_loss_path"]
