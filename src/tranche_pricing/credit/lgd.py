"""Loss-given-default modelling.

We use a Beta distribution on (0, 1) parameterised by mean and standard
deviation. The Beta is the natural choice for an LGD bounded on the unit
interval; the mean / std parameterisation is more readable than the standard
alpha/beta shape parameters when calibrating against survey data on
French rental evictions.

Mean LGD ~ 0.85 matches the typical 14-18 month eviction process net of
Visale guarantees. The standard deviation is calibrated alongside.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import beta as beta_dist


@dataclass(frozen=True, slots=True)
class BetaLGDParams:
    """Beta-LGD parameterised by mean and standard deviation on (0, 1)."""

    mean: float
    std: float

    def __post_init__(self) -> None:
        if not 0 < self.mean < 1:
            raise ValueError("mean must be in (0, 1).")
        if self.std <= 0:
            raise ValueError("std must be > 0.")
        var = self.std**2
        if var >= self.mean * (1 - self.mean):
            raise ValueError(
                f"Beta with mean={self.mean} cannot have std={self.std}: "
                f"variance must be < mean * (1 - mean) = {self.mean * (1 - self.mean):.4f}."
            )

    @property
    def alpha_shape(self) -> float:
        v = self.std**2
        scale = self.mean * (1 - self.mean) / v - 1.0
        return float(self.mean * scale)

    @property
    def beta_shape(self) -> float:
        v = self.std**2
        scale = self.mean * (1 - self.mean) / v - 1.0
        return float((1.0 - self.mean) * scale)


def sample(
    params: BetaLGDParams,
    *,
    n_samples: int,
    rng: np.random.Generator,
) -> NDArray[np.float64]:
    """Draw ``n_samples`` Beta-distributed LGDs."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")
    return rng.beta(params.alpha_shape, params.beta_shape, size=n_samples)


def pdf(params: BetaLGDParams, x: NDArray[np.float64]) -> NDArray[np.float64]:
    """Beta density evaluated at ``x`` (vectorised)."""
    return np.asarray(beta_dist.pdf(x, params.alpha_shape, params.beta_shape), dtype=np.float64)


__all__ = ["BetaLGDParams", "pdf", "sample"]
