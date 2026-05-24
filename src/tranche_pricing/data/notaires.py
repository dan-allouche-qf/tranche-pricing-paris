"""Paris residential price index (Notaires de France / INSEE BDM 001763851).

The Notaires-INSEE quarterly index tracks transaction prices for existing
flats and houses in the Paris area. We use it as the empirical anchor for the
property-price dynamics in :mod:`tranche_pricing.markets.price_gbm` and
:mod:`tranche_pricing.markets.price_jump`.

The public CSV download is the canonical source. The local snapshot at
``data/raw/notaires_paris.csv`` mirrors what was last fetched, so the
calibration pipeline can run offline once the snapshot is in place.
"""

from __future__ import annotations

import logging
from typing import Final

import pandas as pd

from . import _insee_sdmx
from .cache import cached, read_snapshot, record_provenance, write_snapshot

logger = logging.getLogger(__name__)

# INSEE BDM ID 010567013: Indice des prix des logements anciens — Paris —
# Appartements — Base 100 en moyenne annuelle 2015 — Série CVS (seasonally
# adjusted). This is the quarterly Notaires-INSEE index.
SERIES_ID: Final[str] = "010567013"
SERIES_NAME: Final[str] = "notaires_paris"
LABEL: Final[str] = "Paris flats price index (Notaires de France / INSEE, base 100 = 2015)"


@cached
def fetch(
    *,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return the Paris residential price index restricted to ``[start, end]``.

    Parameters
    ----------
    start, end
        Optional ISO-formatted dates used to clip the returned series. ``None``
        keeps the full available history.
    refresh
        When True, the live INSEE CSV is downloaded and the local snapshot is
        overwritten. When False (default), the snapshot is read directly and
        the network is touched only if no snapshot exists yet.
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
    df = df.rename(columns={"value": "price_index"})
    write_snapshot(SERIES_NAME, df.rename(columns={"price_index": "value"}))
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
    # The on-disk snapshot has a generic 'value' column; rename for ergonomics.
    if "value" in df.columns and "price_index" not in df.columns:
        df = df.rename(columns={"value": "price_index"})
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


__all__ = ["LABEL", "SERIES_ID", "SERIES_NAME", "fetch"]
