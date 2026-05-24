"""Calibration sub-package: MLE / GMM fits for every stochastic model.

Each sub-module exposes a ``calibrate`` function with a consistent signature
that returns a :class:`FitResult` containing the fitted parameter dataclass,
asymptotic standard errors (when available) and diagnostic metadata.
"""

from __future__ import annotations

from . import bootstrap, mle_gbm, mle_jump, mle_vasicek
from ._types import FitResult

__all__ = [
    "FitResult",
    "bootstrap",
    "mle_gbm",
    "mle_jump",
    "mle_vasicek",
]
