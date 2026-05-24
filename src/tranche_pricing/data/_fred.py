"""Shared helper for the St. Louis Fed FRED CSV endpoint.

FRED publishes every series as a free CSV at a stable URL that requires no
authentication. The CSV has a fixed two-column header (``observation_date``,
``<series_id>``) and one row per observation in ascending date order.
"""

from __future__ import annotations

import io
import logging
from typing import Final

import pandas as pd

from ._http import UpstreamError, fetch_bytes

logger = logging.getLogger(__name__)

URL_TEMPLATE: Final[str] = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"


def fetch_series(series_id: str) -> pd.DataFrame:
    """Download a FRED series and return a clean ``date, value`` DataFrame."""
    url = URL_TEMPLATE.format(series_id=series_id)
    payload = fetch_bytes(url, headers={"Accept": "text/csv"})
    return _parse(payload, series_id=series_id)


def _parse(payload: bytes, *, series_id: str) -> pd.DataFrame:
    text = payload.decode("utf-8", errors="replace").strip()
    if not text or "<html" in text[:200].lower():
        raise UpstreamError(f"FRED response for {series_id!r} is not CSV.")

    df = pd.read_csv(io.StringIO(text))
    if df.shape[1] != 2:
        raise UpstreamError(
            f"FRED CSV for {series_id!r} has {df.shape[1]} columns; expected 2."
        )

    date_col = df.columns[0]
    value_col = df.columns[1]
    df = df.rename(columns={date_col: "date", value_col: "value"})
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["date", "value"]).sort_values("date").reset_index(drop=True)
    return df


__all__ = ["URL_TEMPLATE", "fetch_series"]
