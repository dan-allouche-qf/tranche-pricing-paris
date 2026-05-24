"""Smoke tests for the visual identity module."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from tranche_pricing.viz.style import (
    CYCLE,
    INSTRUMENT_COLORS,
    PALETTE,
    PLOTLY_TEMPLATE_NAME,
    apply_style,
    mpl_style,
    register_plotly_template,
)


def test_palette_has_expected_keys() -> None:
    required = {"senior", "mezzanine", "equity", "model_a", "model_b", "accent", "neutral"}
    assert required.issubset(PALETTE.keys())
    # All colours must be valid hex.
    for name, hex_code in PALETTE.items():
        assert hex_code.startswith("#"), name
        assert len(hex_code) == 7, name


def test_instrument_colors_pin_each_instrument() -> None:
    assert set(INSTRUMENT_COLORS) == {"model_a", "model_b", "equity", "mezzanine", "senior"}
    # Senior, mezzanine and equity are different colours (no accidental dupes).
    assert len({INSTRUMENT_COLORS[k] for k in INSTRUMENT_COLORS}) == 5


def test_cycle_starts_with_senior_palette() -> None:
    # The colour cycle leads with senior so the first plotted line in a plain
    # plot looks like the senior instrument by default.
    assert CYCLE[0] == PALETTE["senior"]


def test_apply_style_sets_serif_font() -> None:
    apply_style()
    assert plt.rcParams["font.family"] == ["serif"]
    assert plt.rcParams["mathtext.fontset"] == "cm"
    assert plt.rcParams["savefig.dpi"] == 300
    assert plt.rcParams["axes.spines.top"] is False


def test_mpl_style_context_manager_is_scoped() -> None:
    plt.rcParams["axes.spines.top"] = True
    with mpl_style():
        assert plt.rcParams["axes.spines.top"] is False
    assert plt.rcParams["axes.spines.top"] is True
    # restore to the project default for downstream tests
    apply_style()


def test_register_plotly_template_is_idempotent() -> None:
    import plotly.io as pio

    register_plotly_template()
    register_plotly_template()  # calling twice must not raise
    assert pio.templates.default == PLOTLY_TEMPLATE_NAME
