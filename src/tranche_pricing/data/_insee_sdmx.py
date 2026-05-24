"""Shared helper for INSEE BDM (Base de Données Macroéconomiques) SDMX pulls.

INSEE exposes every BDM series as a public SDMX-ML payload at a stable URL
that requires no authentication. The XML carries the metadata (label, source,
frequency, unit) plus the observation list. We parse it into a clean

    date   (pd.Timestamp, period end)
    value  (float)

DataFrame for downstream calibration code.
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Final

import pandas as pd

from ._http import UpstreamError, fetch_bytes

logger = logging.getLogger(__name__)

URL_TEMPLATE: Final[str] = "https://bdm.insee.fr/series/sdmx/data/SERIES_BDM/{series_id}"


def fetch_series(series_id: str) -> pd.DataFrame:
    """Download an INSEE BDM series and parse it into a clean DataFrame."""
    url = URL_TEMPLATE.format(series_id=series_id)
    payload = fetch_bytes(url, headers={"Accept": "application/xml"})
    return _parse(payload, series_id=series_id)


def _parse(payload: bytes, *, series_id: str) -> pd.DataFrame:
    try:
        root = ET.fromstring(payload)
    except ET.ParseError as exc:
        raise UpstreamError(f"INSEE SDMX for series {series_id!r} is not valid XML.") from exc

    observations: list[tuple[pd.Timestamp, float]] = []
    for obs in root.iter():
        # The SDMX-ML compact format puts observations in elements whose local
        # name is "Obs" with TIME_PERIOD and OBS_VALUE attributes.
        if not obs.tag.endswith("}Obs") and obs.tag != "Obs":
            continue
        time_period = obs.get("TIME_PERIOD")
        obs_value = obs.get("OBS_VALUE")
        if time_period is None or obs_value is None:
            continue
        date = _parse_period(time_period)
        if date is None:
            continue
        try:
            value = float(obs_value)
        except ValueError:
            continue
        observations.append((date, value))

    if not observations:
        raise UpstreamError(
            f"INSEE SDMX payload for {series_id!r} did not contain any observation."
        )

    df = pd.DataFrame(observations, columns=["date", "value"])
    return df.sort_values("date").reset_index(drop=True)


def _parse_period(period: str) -> pd.Timestamp | None:
    """Parse the period encodings INSEE uses (YYYY, YYYY-MM, YYYY-Qn, YYYY-Tn)."""
    if not isinstance(period, str):
        return None
    p = period.strip()
    if not p:
        return None

    # Annual: "1996"
    if len(p) == 4 and p.isdigit():
        return pd.Timestamp(year=int(p), month=12, day=31)

    # Quarterly: "1996-Q1" / "1996-T1" / "1996T1" / "1996Q1"
    for sep in ("-T", "T", "-Q", "Q"):
        if sep in p:
            try:
                year_s, q_s = p.split(sep)
                year = int(year_s)
                quarter = int(q_s)
                month_end = {1: 3, 2: 6, 3: 9, 4: 12}[quarter]
                last_day = {3: 31, 6: 30, 9: 30, 12: 31}[month_end]
                return pd.Timestamp(year=year, month=month_end, day=last_day)
            except (ValueError, KeyError):
                continue

    # Monthly: "1996-01" or "1996M01"
    if "-" in p or "M" in p:
        normalised = p.replace("M", "-")
        try:
            ts = pd.Period(normalised, freq="M").to_timestamp(how="end")
            return ts.normalize()
        except (ValueError, TypeError):
            pass

    return None


__all__ = ["URL_TEMPLATE", "fetch_series"]
