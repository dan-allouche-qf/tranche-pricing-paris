"""French 10-year sovereign yield (OAT 10Y).

Sourced from FRED series ``IRLTLT01FRM156N`` ("Long-Term Government Bond
Yields: 10-year: Main (Including Benchmark) for France"). The series is
monthly and goes back to the late 1980s, which is more than enough for the
Vasicek calibration in :mod:`tranche_pricing.markets.rates_vasicek`.

If a higher-frequency / direct-from-source feed is ever needed, the Banque de
France WebStat exposes the same data daily; we keep FRED as the primary
because it is auth-free, encoded as a clean CSV, and reliably retro-revised.
"""

from __future__ import annotations

import logging
from typing import Final

import pandas as pd

from . import _fred
from .cache import cached, read_snapshot, record_provenance, write_snapshot

logger = logging.getLogger(__name__)

SERIES_ID: Final[str] = "IRLTLT01FRM156N"
SERIES_NAME: Final[str] = "oat_10y"
LABEL: Final[str] = "France 10-year sovereign yield (FRED IRLTLT01FRM156N)"


@cached
def fetch(
    *,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return the OAT 10Y yield series (in percent), clipped to ``[start, end]``."""
    df = _load(refresh=refresh)
    return _clip(df, start, end)


def _load(*, refresh: bool) -> pd.DataFrame:
    if not refresh:
        try:
            return read_snapshot(SERIES_NAME)
        except FileNotFoundError:
            logger.info("No %s snapshot found; pulling live.", SERIES_NAME)

    df = _fred.fetch_series(SERIES_ID).rename(columns={"value": "yield_pct"})
    write_snapshot(SERIES_NAME, df.rename(columns={"yield_pct": "value"}))
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
    if "value" in df.columns and "yield_pct" not in df.columns:
        df = df.rename(columns={"value": "yield_pct"})
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


__all__ = ["LABEL", "SERIES_ID", "SERIES_NAME", "fetch"]
