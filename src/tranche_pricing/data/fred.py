"""US S&P / Case-Shiller national home price index (FRED ``CSUSHPISA``).

Used as an out-of-sample sanity check on the price-dynamics calibration of
:mod:`tranche_pricing.markets.price_gbm`. The headline empirical evidence in
the working paper is still Paris (Notaires); Case-Shiller is a robustness
overlay only.
"""

from __future__ import annotations

import logging
from typing import Final

import pandas as pd

from . import _fred
from .cache import cached, read_snapshot, record_provenance, write_snapshot

logger = logging.getLogger(__name__)

SERIES_ID: Final[str] = "CSUSHPISA"
SERIES_NAME: Final[str] = "case_shiller_us"
LABEL: Final[str] = "US S&P / Case-Shiller national home price index (FRED CSUSHPISA)"


@cached
def fetch(
    *,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return the Case-Shiller national index clipped to ``[start, end]``."""
    df = _load(refresh=refresh)
    return _clip(df, start, end)


def _load(*, refresh: bool) -> pd.DataFrame:
    if not refresh:
        try:
            return read_snapshot(SERIES_NAME)
        except FileNotFoundError:
            logger.info("No %s snapshot found; pulling live.", SERIES_NAME)

    df = _fred.fetch_series(SERIES_ID).rename(columns={"value": "price_index"})
    write_snapshot(SERIES_NAME, df.rename(columns={"price_index": "value"}))
    record_provenance(
        SERIES_NAME,
        url=_fred.URL_TEMPLATE.format(series_id=SERIES_ID),
        rows=len(df),
        start=str(df["date"].min().date()) if not df.empty else None,
        end=str(df["date"].max().date()) if not df.empty else None,
        extra={"label": LABEL},
    )
    return df


def _clip(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if "value" in df.columns and "price_index" not in df.columns:
        df = df.rename(columns={"value": "price_index"})
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


__all__ = ["LABEL", "SERIES_ID", "SERIES_NAME", "fetch"]
