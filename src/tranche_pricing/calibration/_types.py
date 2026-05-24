"""Shared types for the calibration layer."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar

T = TypeVar("T")


@dataclass(slots=True)
class FitResult(Generic[T]):
    """Output of an MLE / GMM fit.

    Attributes
    ----------
    params
        Fitted parameter dataclass (e.g. ``GBMParams``, ``VasicekParams``).
    std_errors
        Same shape as ``params``: a mapping of parameter name to standard
        error. ``None`` when not estimated (e.g. by-construction MLE).
    log_likelihood
        Maximum log-likelihood at the optimum.
    n_obs
        Number of observations used in the fit.
    extra
        Free-form diagnostics (convergence flag, optimiser iterations, AIC,
        BIC, etc.).
    """

    params: T
    log_likelihood: float
    n_obs: int
    std_errors: dict[str, float] | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_params(self) -> int:
        if hasattr(self.params, "__dataclass_fields__"):
            return len(self.params.__dataclass_fields__)
        if isinstance(self.params, dict):
            return len(self.params)
        return 0

    @property
    def aic(self) -> float:
        return 2.0 * self.n_params - 2.0 * self.log_likelihood

    @property
    def bic(self) -> float:
        import math

        return self.n_params * math.log(max(self.n_obs, 1)) - 2.0 * self.log_likelihood


__all__ = ["FitResult"]
