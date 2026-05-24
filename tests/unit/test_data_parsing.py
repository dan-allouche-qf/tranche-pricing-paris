"""Parsing tests for the data-acquisition helpers.

We test each parser directly on small fixture bytes that mimic the real
upstream response shape. The HTTP layer is never touched.
"""

from __future__ import annotations

import pytest

from tranche_pricing.data import _fred, _insee_sdmx
from tranche_pricing.data._http import UpstreamError

# --------------------------------------------------------------------------- #
# INSEE SDMX
# --------------------------------------------------------------------------- #

# A minimal SDMX-ML payload that mimics the relevant structure of an INSEE
# response. We only need the elements the parser actually reads.
_INSEE_PAYLOAD = (
    b"<?xml version='1.0' encoding='UTF-8'?>"
    b'<message:StructureSpecificData xmlns:message="urn:m" xmlns:ns="urn:n">'
    b"<message:DataSet>"
    b'<Series IDBANK="010567013">'
    b'<Obs TIME_PERIOD="1996-Q1" OBS_VALUE="100.0"/>'
    b'<Obs TIME_PERIOD="1996-Q2" OBS_VALUE="101.3"/>'
    b'<Obs TIME_PERIOD="1996-Q3" OBS_VALUE="102.5"/>'
    b'<Obs TIME_PERIOD="1996-Q4" OBS_VALUE="104.1"/>'
    b'<Obs TIME_PERIOD="1997-Q1" OBS_VALUE="106.2"/>'
    b"</Series></message:DataSet></message:StructureSpecificData>"
)


def test_insee_sdmx_parses_quarterly_dates() -> None:
    df = _insee_sdmx._parse(_INSEE_PAYLOAD, series_id="010567013")
    assert len(df) == 5
    assert list(df.columns) == ["date", "value"]
    # First row is 1996-Q1 → end of March 1996
    assert df["date"].iloc[0].year == 1996
    assert df["date"].iloc[0].month == 3
    assert df["value"].iloc[0] == pytest.approx(100.0)
    # Last row is 1997-Q1
    assert df["date"].iloc[-1].year == 1997
    assert df["date"].iloc[-1].month == 3


def test_insee_sdmx_parses_monthly_dates() -> None:
    payload = (
        b"<?xml version='1.0' encoding='UTF-8'?>"
        b'<message:StructureSpecificData xmlns:message="urn:m">'
        b'<Series IDBANK="123">'
        b'<Obs TIME_PERIOD="2020-01" OBS_VALUE="128.7"/>'
        b'<Obs TIME_PERIOD="2020-02" OBS_VALUE="129.1"/>'
        b'<Obs TIME_PERIOD="2020-03" OBS_VALUE="129.5"/>'
        b"</Series></message:StructureSpecificData>"
    )
    df = _insee_sdmx._parse(payload, series_id="123")
    assert len(df) == 3
    # End-of-month convention
    assert df["date"].iloc[0].month == 1
    assert df["date"].iloc[0].day == 31


def test_insee_sdmx_rejects_invalid_xml() -> None:
    with pytest.raises(UpstreamError, match="not valid XML"):
        _insee_sdmx._parse(b"<<<not xml>>>", series_id="x")


def test_insee_sdmx_rejects_payload_with_no_observations() -> None:
    payload = (
        b"<?xml version='1.0' encoding='UTF-8'?>"
        b"<message:StructureSpecificData "
        b'xmlns:message="urn:m"><message:DataSet/></message:StructureSpecificData>'
    )
    with pytest.raises(UpstreamError, match="any observation"):
        _insee_sdmx._parse(payload, series_id="x")


def test_insee_sdmx_skips_malformed_observations() -> None:
    payload = (
        b"<?xml version='1.0' encoding='UTF-8'?>"
        b'<message:StructureSpecificData xmlns:message="urn:m">'
        b"<message:DataSet>"
        b'<Series IDBANK="x">'
        b'<Obs TIME_PERIOD="2020-Q1" OBS_VALUE="100.0"/>'
        b'<Obs TIME_PERIOD="bad-period" OBS_VALUE="200.0"/>'
        b'<Obs TIME_PERIOD="2020-Q2" OBS_VALUE="not-a-number"/>'
        b'<Obs TIME_PERIOD="2020-Q3" OBS_VALUE="101.0"/>'
        b"</Series></message:DataSet></message:StructureSpecificData>"
    )
    df = _insee_sdmx._parse(payload, series_id="x")
    # Only the two well-formed obs remain.
    assert len(df) == 2


# --------------------------------------------------------------------------- #
# FRED CSV
# --------------------------------------------------------------------------- #

_FRED_PAYLOAD = (
    b"observation_date,CSUSHPISA\n"
    b"1990-01-01,76.527\n"
    b"1990-02-01,76.703\n"
    b"1990-03-01,76.844\n"
    b"1990-04-01,.\n"  # FRED uses "." for missing
    b"1990-05-01,77.143\n"
)


def test_fred_parses_clean_csv() -> None:
    df = _fred._parse(_FRED_PAYLOAD, series_id="CSUSHPISA")
    # Missing rows are dropped.
    assert len(df) == 4
    assert list(df.columns) == ["date", "value"]
    assert df["date"].iloc[0].year == 1990
    assert df["value"].iloc[0] == pytest.approx(76.527)


def test_fred_rejects_html_responses() -> None:
    html = b"<!DOCTYPE html><html><body>Not found</body></html>"
    with pytest.raises(UpstreamError, match="not CSV"):
        _fred._parse(html, series_id="BAD")


def test_fred_rejects_wrong_column_count() -> None:
    payload = b"date,value,extra\n2020-01-01,1.0,2.0"
    with pytest.raises(UpstreamError, match="columns"):
        _fred._parse(payload, series_id="X")
