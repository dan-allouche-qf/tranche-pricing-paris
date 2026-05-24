"""Quasi-Monte-Carlo helpers.

We replace plain pseudo-random Gaussian draws by inverse-CDF-mapped Sobol
sequences for the most variance-sensitive component of the simulation: the
common factor used by the copula credit models. This typically improves
convergence on Monte Carlo error from O(N^{-1/2}) towards O(N^{-1+eps})
(Owen 1997, Glasserman 2003).

We use :class:`scipy.stats.qmc.Sobol` with random scrambling so the integration
error inherits an unbiased Gaussian sampling theory while keeping the low
discrepancy of the underlying sequence.
"""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm
from scipy.stats import qmc as scipy_qmc


def sobol_gaussian(
    *,
    n_paths: int,
    n_dimensions: int,
    seed: int,
    scramble: bool = True,
) -> NDArray[np.float64]:
    """Generate a Sobol-based Gaussian sample of shape ``(n_paths, n_dimensions)``.

    The Sobol engine draws uniform-in-(0,1)^d points; we transform component-wise
    via the inverse standard-normal CDF, which is the standard way to get a
    QMC Gaussian sample.
    """
    if n_paths <= 0 or n_dimensions <= 0:
        raise ValueError("n_paths and n_dimensions must be positive.")
    engine = scipy_qmc.Sobol(d=n_dimensions, scramble=scramble, seed=seed)
    uniforms = engine.random(n_paths)
    # Clip endpoints to keep the inverse CDF finite.
    uniforms = np.clip(uniforms, a_min=1e-9, a_max=1.0 - 1e-9)
    return cast("NDArray[np.float64]", norm.ppf(uniforms).astype(np.float64))


__all__ = ["sobol_gaussian"]
