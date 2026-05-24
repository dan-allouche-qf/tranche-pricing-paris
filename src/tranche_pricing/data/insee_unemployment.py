"""French unemployment rate (INSEE BDM 001688526).

ILO-definition unemployment rate, seasonally adjusted, quarterly. We use
it as a macro factor proxy for the Cox doubly-stochastic credit model:
rental-default risk is empirically tied to local labour-market slack, and
the unemployment rate is the cleanest publicly available signal.
"""

from __future__ import annotations

import logging
from typing import Final

import pandas as pd

from . import _insee_sdmx
from .cache import cached, read_snapshot, record_provenance, write_snapshot

logger = logging.getLogger(__name__)

SERIES_ID: Final[str] = "001688526"
SERIES_NAME: Final[str] = "insee_unemployment"
LABEL: Final[str] = "France ILO unemployment rate, seasonally adjusted (INSEE)"


@cached
def fetch(
    *,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return the unemployment-rate series (in percent), clipped to ``[start, end]``."""
    df = _load(refresh=refresh)
    return _clip(df, start, end)


def _load(*, refresh: bool) -> pd.DataFrame:
    if not refresh:
        try:
            return read_snapshot(SERIES_NAME)
        except FileNotFoundError:
            logger.info("No %s snapshot found; pulling live.", SERIES_NAME)

    df = _insee_sdmx.fetch_series(SERIES_ID)
    df = df.rename(columns={"value": "unemployment_pct"})
    write_snapshot(SERIES_NAME, df.rename(columns={"unemployment_pct": "value"}))
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
    if "value" in df.columns and "unemployment_pct" not in df.columns:
        df = df.rename(columns={"value": "unemployment_pct"})
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


__all__ = ["LABEL", "SERIES_ID", "SERIES_NAME", "fetch"]
