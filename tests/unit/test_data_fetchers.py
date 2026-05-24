"""Tests for the high-level ``fetch`` functions.

We patch the helper-level network functions so no test ever hits the
internet. The objective here is to verify the fallback / snapshot / clip /
refresh logic, not the URL parsers (those are covered in
``test_data_parsing.py``).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tranche_pricing.data import _fred, _insee_sdmx, fred, insee_irl, notaires, oat, visale
from tranche_pricing.data import cache as cache_module


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr(cache_module, "RAW_DIR", raw)
    monkeypatch.setattr(cache_module, "PROVENANCE_PATH", raw / "_provenance.json")
    # In-memory joblib cache picks up the per-call inputs, but we also flush
    # any state left over from earlier tests in the session.
    cache_module.clear_cache()
    return tmp_path


def _quarterly_index(periods: int = 8, start: str = "1996-01-01") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range(start, periods=periods, freq="QE"),
            "value": [100.0 + i for i in range(periods)],
        }
    )


def _monthly_yield() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.date_range("2002-01-31", periods=24, freq="ME"),
            "value": [3.0 + 0.01 * i for i in range(24)],
        }
    )


def test_notaires_fetch_uses_live_when_snapshot_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def fake_insee(series_id: str) -> pd.DataFrame:
        captured.append(series_id)
        return _quarterly_index()

    monkeypatch.setattr(_insee_sdmx, "fetch_series", fake_insee)
    df = notaires.fetch()
    assert captured == [notaires.SERIES_ID]
    assert "price_index" in df.columns
    # Snapshot was written.
    snapshot_path = cache_module.RAW_DIR / f"{notaires.SERIES_NAME}.csv"
    assert snapshot_path.exists()


def test_notaires_fetch_prefers_snapshot_when_present(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_module.write_snapshot(notaires.SERIES_NAME, _quarterly_index())
    # Live fetcher MUST NOT be called.
    monkeypatch.setattr(
        _insee_sdmx,
        "fetch_series",
        lambda series_id: pytest.fail("live fetch should not happen"),
    )
    df = notaires.fetch()
    assert "price_index" in df.columns
    assert len(df) == 8


def test_notaires_clip_with_start_and_end(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_insee_sdmx, "fetch_series", lambda s: _quarterly_index())
    df = notaires.fetch(start="1996-07-01", end="1997-04-01")
    # That window has 1996-Q3, 1996-Q4 and 1997-Q1.
    assert len(df) == 3
    assert df["date"].min().month in {9}
    assert df["date"].max().month in {3}


def test_notaires_refresh_overwrites_snapshot(monkeypatch: pytest.MonkeyPatch) -> None:
    initial = _quarterly_index(periods=4)
    cache_module.write_snapshot(notaires.SERIES_NAME, initial)
    monkeypatch.setattr(_insee_sdmx, "fetch_series", lambda s: _quarterly_index(periods=10))
    df = notaires.fetch(refresh=True)
    assert len(df) == 10
    reloaded = cache_module.read_snapshot(notaires.SERIES_NAME)
    assert len(reloaded) == 10


def test_insee_irl_fetch_returns_irl_column(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_insee_sdmx, "fetch_series", lambda s: _quarterly_index())
    df = insee_irl.fetch()
    assert "irl" in df.columns


def test_oat_fetch_returns_yield_pct_column(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_fred, "fetch_series", lambda s: _monthly_yield())
    df = oat.fetch()
    assert "yield_pct" in df.columns
    assert df["yield_pct"].iloc[0] == pytest.approx(3.0)


def test_fred_case_shiller_returns_price_index(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_fred, "fetch_series", lambda s: _monthly_yield())
    df = fred.fetch()
    assert "price_index" in df.columns


def test_visale_raises_when_no_snapshot() -> None:
    with pytest.raises(FileNotFoundError, match="No local snapshot"):
        visale.fetch()


def test_visale_reads_committed_snapshot() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2018-12-31", "2019-12-31", "2020-12-31"]),
            "value": [0.028, 0.031, 0.034],
        }
    )
    cache_module.write_snapshot(visale.SERIES_NAME, df)
    out = visale.fetch()
    assert "arrears_rate" in out.columns
    assert len(out) == 3
