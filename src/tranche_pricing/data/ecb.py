"""ECB yield-curve fallback (10-year zero-coupon, euro area AAA government).

The European Central Bank publishes its own euro-area zero-coupon yield curve
at daily frequency under the SDW dataset ``YC``. We pull the 10-year point and
keep it as a fallback / cross-check for the FRED-sourced OAT series in
:mod:`tranche_pricing.data.oat`. The endpoint is auth-free and returns CSV.
"""

from __future__ import annotations

import io
import logging
from typing import Final

import pandas as pd

from ._http import UpstreamError, fetch_bytes
from .cache import cached, read_snapshot, record_provenance, write_snapshot

logger = logging.getLogger(__name__)

# Euro-area AAA government bond, 10Y spot rate, daily.
SDMX_KEY: Final[str] = "B.U2.EUR.4F.G_N_A.SV_C_YM.SR_10Y"
URL: Final[str] = (
    "https://data-api.ecb.europa.eu/service/data/YC/" + SDMX_KEY + "?format=csvdata"
)
SERIES_NAME: Final[str] = "ecb_yield_10y"
LABEL: Final[str] = "Euro-area AAA government 10Y spot rate (ECB SDW)"


@cached
def fetch(
    *,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return the ECB 10Y AAA spot rate (in percent), clipped to ``[start, end]``."""
    df = _load(refresh=refresh)
    return _clip(df, start, end)


def _load(*, refresh: bool) -> pd.DataFrame:
    if not refresh:
        try:
            return read_snapshot(SERIES_NAME)
        except FileNotFoundError:
            logger.info("No %s snapshot found; pulling live.", SERIES_NAME)

    payload = fetch_bytes(URL, headers={"Accept": "text/csv"})
    df = _parse(payload)
    write_snapshot(SERIES_NAME, df.rename(columns={"yield_pct": "value"}))
    record_provenance(
        SERIES_NAME,
        url=URL,
        rows=len(df),
        start=str(df["date"].min().date()) if not df.empty else None,
        end=str(df["date"].max().date()) if not df.empty else None,
        extra={"label": LABEL},
    )
    return df


def _parse(payload: bytes) -> pd.DataFrame:
    raw = pd.read_csv(io.StringIO(payload.decode("utf-8", errors="replace")))
    candidates_date = [c for c in raw.columns if c.upper() in {"TIME_PERIOD", "DATE", "PERIOD"}]
    candidates_value = [c for c in raw.columns if c.upper() in {"OBS_VALUE", "VALUE"}]
    if not candidates_date or not candidates_value:
        raise UpstreamError(
            f"ECB CSV missing expected columns; got {list(raw.columns)!r}"
        )

    df = raw[[candidates_date[0], candidates_value[0]]].copy()
    df.columns = ["date", "yield_pct"]
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["yield_pct"] = pd.to_numeric(df["yield_pct"], errors="coerce")
    return df.dropna().sort_values("date").reset_index(drop=True)


def _clip(df: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    if "value" in df.columns and "yield_pct" not in df.columns:
        df = df.rename(columns={"value": "yield_pct"})
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


__all__ = ["LABEL", "SERIES_NAME", "URL", "fetch"]
