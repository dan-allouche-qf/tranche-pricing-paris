"""Kenneth-French European 3-factor monthly factors.

Used to benchmark the implied risk premium of the tranche structures against
listed European equity factors. Pulled directly from the Tuck-Dartmouth zip
distribution; no auth required.
"""

from __future__ import annotations

import io
import logging
import zipfile
from typing import Final

import pandas as pd

from ._http import UpstreamError, fetch_bytes
from .cache import cached, read_snapshot, record_provenance, write_snapshot

logger = logging.getLogger(__name__)

URL: Final[str] = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "Europe_3_Factors_CSV.zip"
)
SERIES_NAME: Final[str] = "ff_europe_3factors"
LABEL: Final[str] = "Kenneth-French European 3 factors (monthly)"


@cached
def fetch(
    *,
    start: str | None = None,
    end: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Return the monthly Europe 3-factor table, clipped to ``[start, end]``.

    Columns: ``date, mkt_rf, smb, hml, rf`` (all in percent, monthly).
    """
    df = _load(refresh=refresh)
    if start is not None:
        df = df[df["date"] >= pd.Timestamp(start)]
    if end is not None:
        df = df[df["date"] <= pd.Timestamp(end)]
    return df.reset_index(drop=True)


def _load(*, refresh: bool) -> pd.DataFrame:
    if not refresh:
        try:
            return read_snapshot(SERIES_NAME)
        except FileNotFoundError:
            logger.info("No %s snapshot found; pulling live.", SERIES_NAME)

    # The Tuck server rejects narrow Accept headers with 406; use a wildcard.
    payload = fetch_bytes(URL, headers={"Accept": "*/*"})
    df = _parse(payload)
    write_snapshot(SERIES_NAME, df)
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
    with zipfile.ZipFile(io.BytesIO(payload)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise UpstreamError("Fama-French archive does not contain a CSV.")
        with zf.open(names[0]) as fh:
            text = fh.read().decode("utf-8", errors="replace")

    # The Tuck CSV has a fixed structure: title rows, then a header line with
    # column names, then YYYYMM-indexed monthly rows, then an empty line, then
    # annual rows we ignore.
    lines = text.splitlines()
    header_idx = None
    for i, line in enumerate(lines):
        # The header line starts with whitespace before "Mkt-RF".
        if line.lstrip().lower().startswith("mkt-rf"):
            header_idx = i - 0
            break
    if header_idx is None:
        # Fall back: find a line that is a month index "199607" followed by 4 floats.
        for i, line in enumerate(lines):
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 5 and parts[0].isdigit() and len(parts[0]) == 6:
                header_idx = i - 1
                break
    if header_idx is None:
        raise UpstreamError("Could not locate the data block in the Fama-French CSV.")

    # Read monthly block from the header line until the first non-monthly row.
    data_lines = []
    for line in lines[header_idx + 1 :]:
        first = line.split(",")[0].strip()
        if first.isdigit() and len(first) == 6:
            data_lines.append(line)
        elif data_lines:
            # Stop at the first non-monthly row after we have started reading.
            break

    raw_csv = lines[header_idx] + "\n" + "\n".join(data_lines)
    df = pd.read_csv(io.StringIO(raw_csv))
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(
        columns={
            df.columns[0]: "yyyymm",
            "Mkt-RF": "mkt_rf",
            "SMB": "smb",
            "HML": "hml",
            "RF": "rf",
        }
    )
    df["date"] = pd.to_datetime(df["yyyymm"].astype(str) + "01", format="%Y%m%d")
    df["date"] = df["date"] + pd.offsets.MonthEnd(0)
    for col in ("mkt_rf", "smb", "hml", "rf"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df[["date", "mkt_rf", "smb", "hml", "rf"]].dropna().reset_index(drop=True)


__all__ = ["LABEL", "SERIES_NAME", "URL", "fetch"]
