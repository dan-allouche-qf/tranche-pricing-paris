"""Smoke tests for the headline figures."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from tranche_pricing.viz import figures


@pytest.fixture()
def synthetic_notaires() -> pd.DataFrame:
    """A 30-year quarterly synthetic series shaped like the Notaires index.

    Values are deterministic and have nothing to do with real Paris data; the
    test only checks the *plotting* logic.
    """
    dates = pd.date_range("1996-03-31", periods=120, freq="QE")
    rng = np.random.default_rng(42)
    log_path = np.cumsum(rng.normal(loc=0.012, scale=0.015, size=len(dates)))
    return pd.DataFrame({"date": dates, "price_index": 100.0 * np.exp(log_path)})


def test_fig_paris_price_index_returns_figure(synthetic_notaires: pd.DataFrame) -> None:
    fig = figures.fig_paris_price_index(synthetic_notaires)
    assert fig is not None
    # The figure has at least one axes (the main plot) plus optionally the inset.
    assert len(fig.axes) >= 1


def test_fig_paris_price_index_inset_can_be_disabled(
    synthetic_notaires: pd.DataFrame,
) -> None:
    fig = figures.fig_paris_price_index(synthetic_notaires, log_returns_inset=False)
    assert len(fig.axes) == 1


def test_fig_paris_price_index_rejects_missing_column() -> None:
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=4, freq="QE")})
    with pytest.raises(KeyError, match="price_index"):
        figures.fig_paris_price_index(df)


def test_fig_paris_price_index_rejects_empty() -> None:
    df = pd.DataFrame({"date": pd.to_datetime([]), "price_index": pd.Series(dtype="float64")})
    with pytest.raises(ValueError, match="Empty"):
        figures.fig_paris_price_index(df)


def test_fig_paris_price_index_saves_to_pdf(
    synthetic_notaires: pd.DataFrame, tmp_path: pytest.TempPathFactory
) -> None:
    fig = figures.fig_paris_price_index(synthetic_notaires)
    out = tmp_path / "paris.pdf"
    fig.savefig(out)
    assert out.exists()
    assert out.stat().st_size > 1024  # >1 KB indicates real vector content.
