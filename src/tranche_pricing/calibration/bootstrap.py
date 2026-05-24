"""Block-bootstrap helpers for calibration confidence intervals.

A single MLE fit gives a point estimate plus, for the closed-form models,
asymptotic standard errors. For everything else we resample the residual /
log-return series with replacement and refit. The block bootstrap preserves
the short-run autocorrelation structure that an i.i.d. resample would
destroy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

T = TypeVar("T")


@dataclass(slots=True)
class BootstrapResult:
    """Empirical bootstrap distribution of one or more fitted scalars."""

    samples: dict[str, NDArray[np.float64]]
    n_resamples: int

    def quantiles(self, alpha: float = 0.05) -> dict[str, tuple[float, float]]:
        """Return ``(lower, upper)`` quantiles at ``(alpha/2, 1 - alpha/2)``."""
        q = [alpha / 2.0, 1.0 - alpha / 2.0]
        return {
            name: (float(np.quantile(values, q[0])), float(np.quantile(values, q[1])))
            for name, values in self.samples.items()
        }

    def std_errors(self) -> dict[str, float]:
        return {name: float(np.std(values, ddof=1)) for name, values in self.samples.items()}


def block_bootstrap_indices(
    n: int,
    block_size: int,
    n_resamples: int,
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Sample ``n_resamples`` index arrays of length ``n`` by moving-block bootstrap."""
    if block_size <= 0 or n_resamples <= 0 or n <= 0:
        raise ValueError("n, block_size and n_resamples must all be positive.")
    n_blocks = int(np.ceil(n / block_size))
    starts = rng.integers(low=0, high=max(1, n - block_size + 1), size=(n_resamples, n_blocks))
    offsets = np.arange(block_size, dtype=np.int_)[None, None, :]
    indexed = starts[..., None] + offsets
    flat = indexed.reshape(n_resamples, -1)
    return flat[:, :n]


def bootstrap(
    series: NDArray[np.float64],
    fit_fn: Callable[[NDArray[np.float64]], dict[str, float]],
    *,
    n_resamples: int = 1000,
    block_size: int = 1,
    seed: int = 20260519,
) -> BootstrapResult:
    """Run a (moving-block) bootstrap and collect named scalar statistics.

    Parameters
    ----------
    series
        Sample to resample (1-D array; rows are observations).
    fit_fn
        Function taking a resampled array and returning a dict of named
        scalars (typically one entry per fitted parameter).
    n_resamples
        Number of bootstrap iterations.
    block_size
        Length of each bootstrap block. ``1`` is the i.i.d. bootstrap;
        larger values preserve short-run autocorrelation.
    seed
        Seed for the bootstrap RNG (independent from the main MC seed).
    """
    rng = np.random.default_rng(seed)
    indices = block_bootstrap_indices(
        n=len(series), block_size=block_size, n_resamples=n_resamples, rng=rng
    )

    samples: dict[str, list[float]] = {}
    for k in range(n_resamples):
        try:
            stat = fit_fn(series[indices[k]])
        except Exception:  # pragma: no cover - drop failed resamples
            continue
        for name, value in stat.items():
            samples.setdefault(name, []).append(float(value))

    arr_samples = {name: np.array(vals, dtype=float) for name, vals in samples.items()}
    return BootstrapResult(samples=arr_samples, n_resamples=n_resamples)


__all__ = ["BootstrapResult", "block_bootstrap_indices", "bootstrap"]
