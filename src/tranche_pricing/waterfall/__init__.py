"""CDO-style waterfall machinery for the tranche pricing engine.

Public surface:

* :class:`tranches.Tranche` and :func:`tranches.loss_to_tranche`,
  :func:`tranches.allocate_loss_across_stack` — primitives.
* :func:`loss_paths.cumulative_loss_path` — convert credit-layer output to
  cumulative loss path.
* :func:`andersen_sidenius.run` — end-to-end waterfall over the MC batch.
"""

from __future__ import annotations

from . import andersen_sidenius, loss_paths, tranches
from .andersen_sidenius import WaterfallOutcome
from .tranches import Tranche, allocate_loss_across_stack, loss_to_tranche

__all__ = [
    "Tranche",
    "WaterfallOutcome",
    "allocate_loss_across_stack",
    "andersen_sidenius",
    "loss_paths",
    "loss_to_tranche",
    "tranches",
]
