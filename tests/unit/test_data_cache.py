"""Tests for the cache / provenance layer."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from tranche_pricing.data import cache as cache_module


@pytest.fixture()
def isolated_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """Redirect the cache / raw / provenance paths to a per-test tmp_path."""
    raw = tmp_path / "raw"
    raw.mkdir()
    monkeypatch.setattr(cache_module, "RAW_DIR", raw)
    monkeypatch.setattr(cache_module, "PROVENANCE_PATH", raw / "_provenance.json")
    return tmp_path


def _toy_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-31", "2020-06-30", "2020-09-30"]),
            "value": [100.0, 101.0, 102.5],
        }
    )


def test_snapshot_roundtrip_preserves_data(isolated_cache: Path) -> None:
    df = _toy_df()
    path = cache_module.write_snapshot("toy", df)
    assert path.exists()
    reloaded = cache_module.read_snapshot("toy")
    pd.testing.assert_frame_equal(df.reset_index(drop=True), reloaded.reset_index(drop=True))


def test_read_snapshot_raises_when_missing(isolated_cache: Path) -> None:
    with pytest.raises(FileNotFoundError, match="No local snapshot"):
        cache_module.read_snapshot("does_not_exist")


def test_record_and_get_provenance_roundtrip(isolated_cache: Path) -> None:
    cache_module.record_provenance(
        "toy",
        url="https://example.com/x.csv",
        rows=3,
        start="2020-03-31",
        end="2020-09-30",
        extra={"label": "Toy series"},
    )
    record = cache_module.get_provenance("toy")
    assert record is not None
    assert record["url"] == "https://example.com/x.csv"
    assert record["rows"] == 3
    assert record["label"] == "Toy series"

    # On-disk file is valid JSON.
    with cache_module.PROVENANCE_PATH.open() as fh:
        loaded = json.load(fh)
    assert "toy" in loaded


def test_hash_dataframe_stable() -> None:
    h1 = cache_module.hash_dataframe(_toy_df())
    h2 = cache_module.hash_dataframe(_toy_df())
    assert h1 == h2
    assert len(h1) == 16
