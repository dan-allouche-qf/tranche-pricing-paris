"""Project-wide visual identity.

Every figure in the working paper, the notebooks and the dashboard goes through
``apply_style`` so the matplotlib rcParams, the colour palette and the Plotly
template stay perfectly consistent. Importing this module has no side effect by
itself; call ``apply_style()`` once at session start (or rely on the
``mpl_style`` context manager for a scoped override).

The palette is the Okabe-Ito set [Okabe & Ito, 2008] — colour-blind safe and
perceptually distinct in print. Instrument colours are pinned so the same
tranche / model is always rendered the same way across artefacts.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from cycler import cycler

# --------------------------------------------------------------------------- #
# Palette
# --------------------------------------------------------------------------- #

PALETTE: dict[str, str] = {
    "senior": "#0072B2",
    "mezzanine": "#E69F00",
    "equity": "#D55E00",
    "model_a": "#009E73",
    "model_b": "#56B4E9",
    "accent": "#CC79A7",
    "neutral": "#444444",
    "neutral_light": "#BBBBBB",
    "highlight": "#F0E442",
}
"""Project-wide named colours (Okabe-Ito, colour-blind safe)."""

INSTRUMENT_COLORS: dict[str, str] = {
    "model_a": PALETTE["model_a"],
    "model_b": PALETTE["model_b"],
    "equity": PALETTE["equity"],
    "mezzanine": PALETTE["mezzanine"],
    "senior": PALETTE["senior"],
}
"""Stable colour for each instrument in the comparison set."""

CREDIT_MODEL_COLORS: dict[str, str] = {
    "gaussian_copula": PALETTE["neutral"],
    "student_t_copula": PALETTE["accent"],
    "cox_intensity": PALETTE["senior"],
}
"""Stable colour for each credit-risk model in overlays / comparisons."""

STRESS_COLORS: dict[str, str] = {
    "baseline": PALETTE["neutral"],
    "stress_gfc": PALETTE["equity"],
    "stress_covid": PALETTE["mezzanine"],
    "stress_rates2022": PALETTE["senior"],
}

CYCLE = [
    PALETTE["senior"],
    PALETTE["mezzanine"],
    PALETTE["equity"],
    PALETTE["model_a"],
    PALETTE["model_b"],
    PALETTE["accent"],
    PALETTE["neutral"],
]
"""Default colour cycle used when no explicit instrument colour is supplied."""


# --------------------------------------------------------------------------- #
# Matplotlib style
# --------------------------------------------------------------------------- #

# Computer Modern Roman matches the LaTeX serif body text. We fall back to
# whatever serif fonts the host actually carries; on macOS this is typically
# "DejaVu Serif" or "Times New Roman", both of which look acceptable.
_PREFERRED_SERIFS = [
    "Computer Modern Roman",
    "CMU Serif",
    "Latin Modern Roman",
    "STIXGeneral",
    "DejaVu Serif",
    "Times New Roman",
    "Times",
]


def _resolve_serif() -> list[str]:
    """Return the list of serif fonts that actually exist on this machine."""
    available = {f.name for f in fm.fontManager.ttflist}
    resolved = [name for name in _PREFERRED_SERIFS if name in available]
    if not resolved:
        # last resort: ship whatever matplotlib found first
        resolved = ["DejaVu Serif"]
    return resolved


RCPARAMS: dict[str, object] = {
    # Typography ------------------------------------------------------------ #
    "font.family": "serif",
    "font.serif": _resolve_serif(),
    "mathtext.fontset": "cm",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 13,
    "figure.titleweight": "bold",
    # Figure ---------------------------------------------------------------- #
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.04,
    "figure.figsize": (6.5, 4.0),
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    # Axes ----------------------------------------------------------------- #
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#222222",
    "axes.labelcolor": "#222222",
    "axes.titlepad": 10.0,
    "axes.prop_cycle": cycler(color=CYCLE),
    # Grid ----------------------------------------------------------------- #
    "axes.grid": True,
    "axes.grid.axis": "y",
    "axes.grid.which": "major",
    "grid.color": "#BBBBBB",
    "grid.alpha": 0.28,
    "grid.linestyle": "-",
    "grid.linewidth": 0.5,
    # Ticks ---------------------------------------------------------------- #
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    # Legend --------------------------------------------------------------- #
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "legend.borderaxespad": 0.4,
    "legend.columnspacing": 1.2,
    # PDF / PGF ------------------------------------------------------------- #
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def apply_style() -> None:
    """Apply the project-wide matplotlib rcParams in place."""
    mpl.rcParams.update(RCPARAMS)


@contextmanager
def mpl_style() -> Iterator[None]:
    """Context manager that scopes the project rcParams to a ``with`` block."""
    with plt.rc_context(RCPARAMS):
        yield


# --------------------------------------------------------------------------- #
# Plotly template (used by the Streamlit dashboard)
# --------------------------------------------------------------------------- #

PLOTLY_TEMPLATE_NAME = "tranche_paper"


def register_plotly_template() -> None:
    """Register the Plotly template once so the dashboard pages share style.

    The template is intentionally minimal: serif font, neutral grid, the same
    colour cycle as matplotlib, no top/right "spines" (Plotly draws axes on
    every side by default, we hide them for parity).
    """
    try:
        import plotly.graph_objects as go
        import plotly.io as pio
    except ImportError:  # pragma: no cover
        return

    template = go.layout.Template()
    template.layout = go.Layout(
        font={"family": "serif", "size": 12, "color": "#222222"},
        title={"font": {"size": 14}},
        plot_bgcolor="white",
        paper_bgcolor="white",
        colorway=CYCLE,
        xaxis={
            "showgrid": False,
            "zeroline": False,
            "showline": True,
            "linecolor": "#222222",
            "linewidth": 0.8,
            "ticks": "outside",
            "tickcolor": "#222222",
        },
        yaxis={
            "showgrid": True,
            "gridcolor": "#BBBBBB",
            "zeroline": False,
            "showline": True,
            "linecolor": "#222222",
            "linewidth": 0.8,
            "ticks": "outside",
            "tickcolor": "#222222",
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0.0},
    )
    pio.templates[PLOTLY_TEMPLATE_NAME] = template
    pio.templates.default = PLOTLY_TEMPLATE_NAME


__all__ = [
    "CREDIT_MODEL_COLORS",
    "CYCLE",
    "INSTRUMENT_COLORS",
    "PALETTE",
    "PLOTLY_TEMPLATE_NAME",
    "RCPARAMS",
    "STRESS_COLORS",
    "apply_style",
    "mpl_style",
    "register_plotly_template",
]
