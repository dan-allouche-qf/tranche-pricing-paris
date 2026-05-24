"""Deterministic per-stream RNG seeding.

Every random source in the simulation engine is decorrelated by deriving its
RNG from a single ``master_seed`` via :class:`numpy.random.SeedSequence`. This
gives bit-identical reproducibility across runs while keeping the streams
independent, which is what variance-reduction techniques and convergence
diagnostics need.

The named streams are: ``price``, ``rate``, ``credit``, ``lgd``, ``qmc``.
Adding a new stream is as cheap as adding a name to :data:`STREAM_NAMES`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

STREAM_NAMES: Final[tuple[str, ...]] = ("price", "rate", "credit", "lgd", "qmc")


@dataclass(frozen=True, slots=True)
class StreamRNGs:
    """Bundle of named ``numpy.random.Generator`` instances."""

    price: np.random.Generator
    rate: np.random.Generator
    credit: np.random.Generator
    lgd: np.random.Generator
    qmc: np.random.Generator

    def as_dict(self) -> dict[str, np.random.Generator]:
        return {name: getattr(self, name) for name in STREAM_NAMES}


def make_streams(master_seed: int) -> StreamRNGs:
    """Derive one independent ``Generator`` per named stream from a master seed."""
    sequence = np.random.SeedSequence(master_seed)
    children = sequence.spawn(len(STREAM_NAMES))
    return StreamRNGs(
        **dict(zip(STREAM_NAMES, [np.random.default_rng(c) for c in children], strict=False))
    )


__all__ = ["STREAM_NAMES", "StreamRNGs", "make_streams"]
