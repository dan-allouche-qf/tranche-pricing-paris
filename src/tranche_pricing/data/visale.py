"""Visale / ANIL rental-arrears snapshot (local CSV only).

There is no programmatic feed for the Action Logement / Visale claims report;
it is published yearly as a PDF table at
https://www.actionlogement.fr/sites/default/files/visale-rapport-annuel.pdf
and the ANIL barometer at https://www.anil.org/. We rely on a manual CSV in
``data/raw/visale.csv`` containing the yearly arrears rate that the user
extracts from those public sources and commits to the repo. The expected
schema is::

    date,value
    2018-12-31,0.029
    2019-12-31,0.031
    ...

The ``value`` column is the share of leases in arrears (decimal, not percent).
"""

from __future__ import annotations

import logging
from typing import Final

import pandas as pd

from .cache import cached, read_snapshot

logger = logging.getLogger(__name__)

SERIES_NAME: Final[str] = "visale"
LABEL: Final[str] = "Visale / ANIL rental-arrears rate (manual snapshot)"


@cached
def fetch(
    *,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,  # accepted for API parity; ignored here
) -> pd.DataFrame:
    """Read the local Visale snapshot and clip to ``[start, end]``.

    Raises
    ------
    FileNotFoundError
        If ``data/raw/visale.csv`` has not been populated yet. The error
        message points the user to ``data/DATA_SOURCES.md``.
    """
    del refresh
    df = read_snapshot(SERIES_NAME)
    df = df.rename(columns={"value": "arrears_rate"})
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


__all__ = ["LABEL", "SERIES_NAME", "fetch"]
