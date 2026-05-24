"""Indice de Référence des Loyers (INSEE BDM 001515333).

The IRL is the legal rent-indexation reference in France: published quarterly,
all residential leases are repriced against this index. We use it to drive
rent indexation in :mod:`tranche_pricing.markets` and to construct a
default-stress proxy for the credit layer.
"""

from __future__ import annotations

import logging
from typing import Final

import pandas as pd

from . import _insee_sdmx
from .cache import cached, read_snapshot, record_provenance, write_snapshot

logger = logging.getLogger(__name__)

SERIES_ID: Final[str] = "001515333"
SERIES_NAME: Final[str] = "insee_irl"
LABEL: Final[str] = "Indice de Référence des Loyers (INSEE)"


@cached
def fetch(
    *,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return the IRL series clipped to ``[start, end]``.

    See :func:`tranche_pricing.data.notaires.fetch` for parameter semantics.
    """
    df = _load(refresh=refresh)
    return _clip(df, start, end)


def _load(*, refresh: bool) -> pd.DataFrame:
    if not refresh:
        try:
            return read_snapshot(SERIES_NAME)
        except FileNotFoundError:
            logger.info("No %s snapshot found; pulling live.", SERIES_NAME)

    df = _insee_sdmx.fetch_series(SERIES_ID)
    df = df.rename(columns={"value": "irl"})
    write_snapshot(SERIES_NAME, df.rename(columns={"irl": "value"}))
    record_provenance(
        SERIES_NAME,
        url=_insee_sdmx.URL_TEMPLATE.format(series_id=SERIES_ID),
        rows=len(df),
        start=str(df["date"].min().date()) if not df.empty else None,
        end=str(df["date"].max().date()) if not df.empty else None,
        extra={"label": LABEL},
    )
    return df


def _clip(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if "value" in df.columns and "irl" not in df.columns:
        df = df.rename(columns={"value": "irl"})
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


__all__ = ["LABEL", "SERIES_ID", "SERIES_NAME", "fetch"]
