"""Headline figures for the working paper and the dashboard.

Each function in this module produces one figure from the working paper's
inventory (see ``docs/`` and the plan in
``regarde-j-ai-fait-ce-playful-avalanche.md``). They all share the same
shape::

    fig = fig_<name>(data, **options)
    fig.savefig(...)

Functions never fetch data themselves: the caller (notebook, CLI, dashboard)
is responsible for handing in a cleanly-loaded DataFrame. This keeps the
figures testable on small fixtures and makes them trivial to reuse.

When called with the project style applied via :func:`tranche_pricing.viz.
style.apply_style`, every figure inherits the serif typography, the
Okabe-Ito palette and the layout choices documented there.
"""

from __future__ import annotations

import logging
from typing import Final

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .style import PALETTE, apply_style

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Recession shading (used by several time-series figures)                     #
# --------------------------------------------------------------------------- #

# French / euro-area recession episodes used for visual context. The dates are
# trough-to-peak inclusive and follow the OECD CLI-based reference dating for
# France (long version), plus the COVID shock which is uncontested. Labels
# are short to avoid overlap when several bands sit near each other.
FRENCH_RECESSIONS: Final[list[tuple[str, str, str]]] = [
    ("2008-01-31", "2009-06-30", "GFC"),
    ("2011-09-30", "2013-03-31", "Euro debt"),
    ("2020-02-29", "2020-05-31", "COVID"),
    ("2022-04-30", "2023-09-30", "Rate hike"),
]


def _shade_recessions(
    ax: plt.Axes,
    episodes: list[tuple[str, str, str]] | None = None,
    *,
    label: bool = True,
) -> None:
    """Overlay translucent vertical bands for each recession episode.

    Labels alternate between two vertical positions so they do not overlap
    when bands are close together (e.g. COVID and Rate-hike).
    """
    eps = episodes if episodes is not None else FRENCH_RECESSIONS
    ymin, ymax = ax.get_ylim()
    span = ymax - ymin
    high = ymax - 0.02 * span
    low = ymax - 0.10 * span
    for i, (start, end, name) in enumerate(eps):
        ax.axvspan(
            pd.Timestamp(start),
            pd.Timestamp(end),
            color=PALETTE["neutral_light"],
            alpha=0.35,
            linewidth=0,
            zorder=0,
        )
        if label:
            ax.text(
                pd.Timestamp(start) + (pd.Timestamp(end) - pd.Timestamp(start)) / 2,
                high if i % 2 == 0 else low,
                name,
                ha="center",
                va="top",
                fontsize=8,
                color=PALETTE["neutral"],
                alpha=0.85,
                zorder=2,
            )
    ax.set_ylim(ymin, ymax)


# --------------------------------------------------------------------------- #
# Figure #1 — Paris residential price index                                   #
# --------------------------------------------------------------------------- #


def fig_paris_price_index(
    notaires: pd.DataFrame,
    *,
    log_returns_inset: bool = True,
    title: str = "Paris residential price index",
    subtitle: str = "Source: Notaires de France / INSEE (BDM 010567013, base 100 = 2015)",
) -> Figure:
    """Plot the Notaires-INSEE Paris price index with recession shading.

    Parameters
    ----------
    notaires
        DataFrame with columns ``date`` and ``price_index`` (see
        :func:`tranche_pricing.data.notaires.fetch`).
    log_returns_inset
        If True, overlays a small inset showing quarterly log-returns. This is
        what GBM / Merton calibration actually consumes.
    title, subtitle
        Top-of-axes labels. Subtitle is rendered in lighter weight.
    """
    apply_style()

    if "price_index" not in notaires.columns:
        raise KeyError("`notaires` must have a 'price_index' column.")
    df = notaires.dropna(subset=["price_index"]).sort_values("date").reset_index(drop=True)
    if df.empty:
        raise ValueError("Empty notaires DataFrame: cannot plot.")

    fig = plt.figure(figsize=(7.5, 4.6))
    ax = fig.add_subplot(111)

    ax.plot(df["date"], df["price_index"], color=PALETTE["senior"], linewidth=1.4)
    ax.set_xlabel("Date")
    ax.set_ylabel("Price index")
    ax.set_title(title, loc="left", pad=14)
    ax.text(
        0.0,
        1.02,
        subtitle,
        transform=ax.transAxes,
        fontsize=9,
        color=PALETTE["neutral"],
    )
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    _shade_recessions(ax)

    # Last-observed annotation.
    last = df.iloc[-1]
    ax.annotate(
        f"{last['price_index']:.1f}\n{last['date'].strftime('%Y-%m')}",
        xy=(last["date"], last["price_index"]),
        xytext=(8, 0),
        textcoords="offset points",
        va="center",
        ha="left",
        fontsize=9,
        color=PALETTE["senior"],
    )

    fig.tight_layout()

    if log_returns_inset and len(df) > 8:
        log_ret = np.log(df["price_index"]).diff().dropna() * 100  # in %
        inset = fig.add_axes((0.62, 0.22, 0.32, 0.22))
        inset.plot(df["date"].iloc[1:], log_ret, color=PALETTE["accent"], linewidth=0.7)
        inset.set_title(
            "Quarterly log-returns (%)", fontsize=8, loc="left", color=PALETTE["neutral"]
        )
        inset.tick_params(axis="both", which="major", labelsize=7)
        inset.xaxis.set_major_locator(mdates.YearLocator(10))
        inset.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        inset.spines["top"].set_visible(False)
        inset.spines["right"].set_visible(False)
        inset.axhline(0, color=PALETTE["neutral"], linewidth=0.5, alpha=0.5)

    return fig


# --------------------------------------------------------------------------- #
# Figure #13 — Calibration diagnostics                                        #
# --------------------------------------------------------------------------- #


def fig_calibration_diagnostics(
    *,
    paris_log_returns: pd.Series,
    gbm_sigma: float,
    gbm_dt: float,
    oat_residuals: pd.Series,
    oat_sigma_eps: float,
    max_lag: int = 20,
) -> Figure:
    """Four-panel diagnostic for the market calibrations.

    Layout (2 x 2):
        (top-left)  Histogram of Paris quarterly log-returns + fitted normal.
        (top-right) QQ-plot of Paris log-returns vs N(0, sigma_GBM^2 * dt).
        (bot-left)  Vasicek AR(1) residuals histogram + fitted normal.
        (bot-right) Auto-correlation of squared Paris log-returns up to max_lag.

    A well-fitted GBM has a roughly straight QQ-plot. Persistent ACF in
    squared returns is the volatility-clustering signature that motivates the
    Merton-jump or stochastic-volatility extensions.
    """
    from scipy.stats import norm

    apply_style()

    r_paris = paris_log_returns.dropna().to_numpy(dtype=float)
    res_oat = oat_residuals.dropna().to_numpy(dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(9.5, 6.5))
    fig.suptitle(
        "Calibration diagnostics (Notaires Paris GBM, OAT 10Y Vasicek)",
        fontsize=11,
        x=0.04,
        ha="left",
        y=0.995,
    )

    # (A) Paris log-returns histogram with fitted normal
    ax = axes[0, 0]
    bins = np.linspace(r_paris.min() - 0.005, r_paris.max() + 0.005, 25)
    ax.hist(
        r_paris, bins=bins, density=True, color=PALETTE["senior"], alpha=0.55, edgecolor="white"
    )
    grid = np.linspace(bins[0], bins[-1], 400)
    sigma_q = gbm_sigma * np.sqrt(gbm_dt)
    mean_q = r_paris.mean()
    ax.plot(
        grid,
        norm.pdf(grid, loc=mean_q, scale=sigma_q),
        color=PALETTE["accent"],
        linewidth=1.3,
        label=r"$\mathcal{N}(\bar r,\,\sigma_{\rm GBM}\sqrt{\Delta t}^2)$",
    )
    ax.set_xlabel("Quarterly log-return")
    ax.set_ylabel("Density")
    ax.set_title("(a) Paris log-returns vs fitted Gaussian", loc="left", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)

    # (B) QQ-plot
    ax = axes[0, 1]
    sorted_r = np.sort(r_paris)
    n = sorted_r.size
    plotting_positions = (np.arange(1, n + 1) - 0.5) / n
    theoretical = norm.ppf(plotting_positions, loc=mean_q, scale=sigma_q)
    ax.scatter(theoretical, sorted_r, s=10, color=PALETTE["senior"], alpha=0.7)
    diag_min = min(theoretical.min(), sorted_r.min())
    diag_max = max(theoretical.max(), sorted_r.max())
    ax.plot([diag_min, diag_max], [diag_min, diag_max], color=PALETTE["neutral"], linewidth=0.8)
    ax.set_xlabel("Theoretical N quantile")
    ax.set_ylabel("Empirical quantile")
    ax.set_title("(b) QQ-plot: Paris log-returns", loc="left", fontsize=10)

    # (C) Vasicek residuals histogram
    ax = axes[1, 0]
    bins_r = np.linspace(res_oat.min() - 1e-4, res_oat.max() + 1e-4, 30)
    ax.hist(
        res_oat,
        bins=bins_r,
        density=True,
        color=PALETTE["mezzanine"],
        alpha=0.55,
        edgecolor="white",
    )
    grid_r = np.linspace(bins_r[0], bins_r[-1], 400)
    ax.plot(
        grid_r,
        norm.pdf(grid_r, loc=0.0, scale=oat_sigma_eps),
        color=PALETTE["accent"],
        linewidth=1.3,
        label=r"$\mathcal{N}(0,\,\sigma_\varepsilon^2)$",
    )
    ax.set_xlabel("OAT AR(1) residual")
    ax.set_ylabel("Density")
    ax.set_title("(c) Vasicek residuals vs fitted Gaussian", loc="left", fontsize=10)
    ax.legend(loc="upper left", fontsize=8)

    # (D) ACF of squared Paris log-returns
    ax = axes[1, 1]
    lags = np.arange(1, max_lag + 1)
    sq = (r_paris - r_paris.mean()) ** 2
    sq_centered = sq - sq.mean()
    denom = float((sq_centered**2).sum())
    acf_vals = np.array(
        [float((sq_centered[:-lag] * sq_centered[lag:]).sum()) / denom for lag in lags]
    )
    ax.bar(lags, acf_vals, color=PALETTE["senior"], width=0.7, edgecolor="white")
    se = 1.96 / np.sqrt(len(r_paris))
    ax.axhline(se, color=PALETTE["accent"], linestyle="--", linewidth=0.8, alpha=0.85)
    ax.axhline(-se, color=PALETTE["accent"], linestyle="--", linewidth=0.8, alpha=0.85)
    ax.axhline(0.0, color=PALETTE["neutral"], linewidth=0.5)
    ax.set_xlabel("Lag (quarters)")
    ax.set_ylabel("ACF of $r^2$")
    ax.set_title("(d) Volatility clustering check (Paris)", loc="left", fontsize=10)

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return fig


# --------------------------------------------------------------------------- #
# Figure #4 — Loss distributions across credit models                         #
# --------------------------------------------------------------------------- #


def fig_loss_distributions(
    *,
    losses_by_model: dict[str, np.ndarray],
    bin_width: float | None = None,
    title: str = "Distribution of cumulative portfolio loss",
    subtitle: str = "Gaussian, Student-t and Cox doubly-stochastic models on the same calibration",
    var_95: bool = True,
) -> Figure:
    """Overlay the loss-fraction histograms produced by each credit model.

    Parameters
    ----------
    losses_by_model
        Mapping ``credit_model_name -> 1D array of cumulative loss fractions``
        (one element per Monte Carlo path, in [0, 1]).
    bin_width
        Histogram bin width in loss-fraction units. Inferred from the data
        when omitted.
    title, subtitle
        Top-of-axes labels.
    var_95
        Draw the 95% VaR for each model as a dashed vertical line.
    """
    from .style import CREDIT_MODEL_COLORS

    apply_style()
    if not losses_by_model:
        raise ValueError("losses_by_model must be non-empty.")

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    all_values = np.concatenate(list(losses_by_model.values()))
    upper = float(np.quantile(all_values, 0.995))
    upper = max(upper, 0.05)
    if bin_width is None:
        bin_width = upper / 40

    bins = list(np.arange(0, upper + bin_width, bin_width))
    for name, values in losses_by_model.items():
        color = CREDIT_MODEL_COLORS.get(name, PALETTE["neutral"])
        ax.hist(
            values,
            bins=bins,
            density=True,
            alpha=0.45,
            color=color,
            edgecolor="white",
            label=name.replace("_", " ").title(),
        )
        if var_95:
            v = float(np.quantile(values, 0.95))
            ax.axvline(v, color=color, linestyle="--", linewidth=1.0, alpha=0.85)
            ax.text(
                v,
                ax.get_ylim()[1] * 0.95,
                f"VaR$_{{95}}$ = {v:.2%}",
                color=color,
                fontsize=8,
                rotation=90,
                va="top",
                ha="right",
            )

    ax.set_xlabel("Cumulative portfolio loss (fraction of par)")
    ax.set_ylabel("Density")
    ax.set_title(title, loc="left", pad=14)
    ax.text(0.0, 1.04, subtitle, transform=ax.transAxes, fontsize=9, color=PALETTE["neutral"])
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_xlim(0, upper)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure #5 — Lower tail dependence of the Student-t copula                   #
# --------------------------------------------------------------------------- #


def fig_tail_dependence(
    *,
    rho_grid: np.ndarray | None = None,
    nu_grid: list[float] | None = None,
    title: str = "Lower tail dependence — Student-t copula",
    subtitle: str = "Closed form from Embrechts–McNeil–Straumann (2002)",
) -> Figure:
    """Plot the lower tail-dependence as a function of rho for several nu.

    The Gaussian limit (``nu -> infty``) sits on the x-axis: no tail
    dependence regardless of ``rho``. As ``nu`` shrinks, tail dependence
    grows substantially — the empirical reason the Gaussian copula
    systematically under-prices senior tranches in stress.
    """
    from ..credit.student_t_copula import tail_dependence_lower
    from .style import CREDIT_MODEL_COLORS

    apply_style()
    rho_grid = np.linspace(0.0, 0.6, 121) if rho_grid is None else np.asarray(rho_grid)
    nu_grid = [3.0, 5.0, 8.0, 30.0] if nu_grid is None else list(nu_grid)

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colour_pool = [
        PALETTE["equity"],
        PALETTE["mezzanine"],
        PALETTE["accent"],
        PALETTE["senior"],
    ]
    for nu, colour in zip(nu_grid, colour_pool[: len(nu_grid)], strict=False):
        vals = [tail_dependence_lower(rho=float(r), nu=float(nu)) for r in rho_grid]
        ax.plot(rho_grid, vals, color=colour, linewidth=1.4, label=rf"$\nu = {nu:g}$")

    # Gaussian baseline at lambda = 0.
    ax.axhline(0.0, color=CREDIT_MODEL_COLORS["gaussian_copula"], linewidth=0.8, linestyle=":")
    ax.text(
        rho_grid[-1] * 0.55,
        0.02,
        r"Gaussian copula ($\lambda_L \equiv 0$)",
        fontsize=9,
        color=CREDIT_MODEL_COLORS["gaussian_copula"],
    )

    ax.set_xlabel(r"Correlation $\rho$")
    ax.set_ylabel(r"Lower tail dependence $\lambda_L$")
    ax.set_xlim(rho_grid[0], rho_grid[-1])
    ax.set_ylim(-0.02, 1.0)
    ax.set_title(title, loc="left", pad=14)
    ax.text(0.0, 1.04, subtitle, transform=ax.transAxes, fontsize=8.5, color=PALETTE["neutral"])
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure #3 — Tranche waterfall explainer                                     #
# --------------------------------------------------------------------------- #


def fig_waterfall_explainer(
    *,
    tranches: list,  # list[Tranche], typed loosely to avoid an import cycle
    grid_resolution: int = 401,
    title: str = "Tranche loss absorption (stop-loss waterfall)",
    subtitle: str = (
        "Stacked loss-allocation curve as the aggregate portfolio loss grows from 0 to 100% of par"
    ),
) -> Figure:
    """Stacked-area plot of per-tranche loss vs aggregate cumulative loss.

    The curve for each tranche is its stop-loss payoff
    ``loss(L) = min(max(L - attach, 0), thickness)`` evaluated on a fine grid
    of aggregate losses ``L in [0, 1]``. Stacking them produces the 45-degree
    line ``y = L``, which makes the loss-absorption order self-evident.
    """
    from ..waterfall.tranches import loss_to_tranche
    from .style import INSTRUMENT_COLORS

    apply_style()

    sorted_tr = sorted(tranches, key=lambda t: t.attach)
    grid = np.linspace(0.0, 1.0, grid_resolution)
    stacks = [np.asarray(loss_to_tranche(grid, t), dtype=float) for t in sorted_tr]

    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    bottoms = np.zeros_like(grid)
    for tr, contrib in zip(sorted_tr, stacks, strict=True):
        ax.fill_between(
            grid,
            bottoms,
            bottoms + contrib,
            color=INSTRUMENT_COLORS.get(tr.name, PALETTE["neutral"]),
            alpha=0.65,
            linewidth=0,
            label=f"{tr.name.title()}  ({tr.attach:.0%}–{tr.detach:.0%})",
        )
        bottoms = bottoms + contrib

    # Annotate the detach points where a tranche becomes exhausted.
    for tr in sorted_tr[:-1]:
        ax.axvline(tr.detach, color=PALETTE["neutral"], linestyle="--", linewidth=0.7, alpha=0.7)
        ax.text(
            tr.detach,
            0.04,
            f"detach\n{tr.detach:.0%}",
            fontsize=8,
            ha="center",
            va="bottom",
            color=PALETTE["neutral"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        )

    ax.plot(grid, grid, color=PALETTE["neutral"], linewidth=0.6, linestyle=":", alpha=0.6)
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 1.0)
    ax.set_xlabel("Aggregate portfolio loss (fraction of par)")
    ax.set_ylabel("Loss absorbed by tranche (fraction of par)")
    ax.set_title(title, loc="left", pad=14)
    ax.text(0.0, 1.04, subtitle, transform=ax.transAxes, fontsize=9, color=PALETTE["neutral"])
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure #7 — Monte Carlo convergence (variance reduction)                    #
# --------------------------------------------------------------------------- #


def fig_mc_convergence(
    *,
    n_paths_grid: list[int],
    se_by_method: dict[str, list[float]],
    title: str = "Monte Carlo convergence",
    subtitle: str = (
        "Standard error of an estimator vs MC sample size — log-log axes; "
        "reference slopes -1/2 (MC) and -1 (QMC)."
    ),
) -> Figure:
    """Plot the empirical SE of an estimator vs the number of MC paths.

    Parameters
    ----------
    n_paths_grid
        Sample sizes used along the x-axis.
    se_by_method
        ``method_name -> [SE_at_each_n]`` — same length as ``n_paths_grid``.
    """
    apply_style()

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colours = [PALETTE["senior"], PALETTE["mezzanine"], PALETTE["equity"], PALETTE["accent"]]

    for (name, ses), colour in zip(se_by_method.items(), colours, strict=False):
        ax.loglog(
            n_paths_grid,
            ses,
            marker="o",
            linewidth=1.3,
            color=colour,
            label=name.replace("_", " ").title(),
        )

    # Reference slopes anchored at the leftmost point.
    n_arr = np.asarray(n_paths_grid, dtype=float)
    anchor = max(ses[0] for ses in se_by_method.values())
    ax.loglog(
        n_arr,
        anchor * (n_arr[0] / n_arr) ** 0.5,
        color=PALETTE["neutral"],
        linestyle="--",
        linewidth=0.8,
        alpha=0.65,
        label=r"$O(N^{-1/2})$",
    )
    ax.loglog(
        n_arr,
        anchor * (n_arr[0] / n_arr),
        color=PALETTE["neutral_light"],
        linestyle=":",
        linewidth=0.8,
        alpha=0.85,
        label=r"$O(N^{-1})$",
    )

    ax.set_xlabel(r"Number of Monte Carlo paths $N$")
    ax.set_ylabel("Standard error of the estimator")
    ax.set_title(title, loc="left", pad=14)
    ax.text(0.0, 1.04, subtitle, transform=ax.transAxes, fontsize=8.5, color=PALETTE["neutral"])
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure #8 — Risk-adjusted metrics bars                                      #
# --------------------------------------------------------------------------- #


def fig_riskreturn_bars(
    *,
    results: pd.DataFrame,
    metric_columns: tuple[str, ...] = ("risk_sharpe", "risk_sortino", "risk_omega", "risk_calmar"),
    credit_model: str = "gaussian_copula",
    insurance: str = "none",
    title: str = "Risk-adjusted metrics across instruments",
    subtitle: str | None = None,
) -> Figure:
    """Side-by-side bars of risk-adjusted ratios for each instrument."""
    from .style import INSTRUMENT_COLORS

    apply_style()
    df = results[(results["credit_model"] == credit_model) & (results["insurance"] == insurance)]
    if df.empty:
        raise ValueError(f"No rows match credit_model={credit_model!r}, insurance={insurance!r}.")

    pretty = {
        "risk_sharpe": "Sharpe",
        "risk_sortino": "Sortino",
        "risk_omega": "Omega",
        "risk_calmar": "Calmar",
    }
    instruments = list(df["instrument"])
    n_inst = len(instruments)
    width = 0.18
    x = np.arange(len(metric_columns))

    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    for i, inst in enumerate(instruments):
        vals = df[df["instrument"] == inst][list(metric_columns)].values.flatten()
        ax.bar(
            x + (i - (n_inst - 1) / 2) * width,
            vals,
            width=width,
            color=INSTRUMENT_COLORS.get(inst, PALETTE["neutral"]),
            label=inst.replace("_", " ").title(),
            edgecolor="white",
            linewidth=0.5,
        )

    ax.axhline(0.0, color=PALETTE["neutral"], linewidth=0.6, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels([pretty.get(c, c) for c in metric_columns])
    ax.set_ylabel("Ratio value")
    ax.set_title(title, loc="left", pad=14)
    sub = (
        subtitle
        if subtitle is not None
        else f"Credit model: {credit_model.replace('_', ' ').title()} · insurance: {insurance}"
    )
    ax.text(0.0, 1.02, sub, transform=ax.transAxes, fontsize=9, color=PALETTE["neutral"])
    ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=n_inst // 2 + 1)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure #9 — Risk-return Pareto frontier                                     #
# --------------------------------------------------------------------------- #


def fig_pareto_frontier(
    *,
    results: pd.DataFrame,
    x_metric: str = "risk_std",
    y_metric: str = "mean_ann_return",
    title: str = "Risk-return frontier across instruments",
    subtitle: str = "Standard deviation vs mean annualised return; markers shaped by insurance",
) -> Figure:
    """Scatter of annualised return vs realised standard deviation per instrument."""
    from .style import INSTRUMENT_COLORS

    apply_style()
    df = results.copy()
    if x_metric not in df.columns:
        raise KeyError(f"x_metric {x_metric!r} not in results columns.")
    if y_metric not in df.columns:
        raise KeyError(f"y_metric {y_metric!r} not in results columns.")

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    markers = {"none": "o", "actuarial": "s", "option_theoretic": "^"}
    seen_combos: set[str] = set()
    for _, row in df.iterrows():
        instrument = row["instrument"]
        insurance = row.get("insurance", "none")
        marker = markers.get(insurance, "x")
        label = f"{instrument} ({insurance})"
        if label in seen_combos:
            continue
        seen_combos.add(label)
        ax.scatter(
            row[x_metric],
            row[y_metric],
            color=INSTRUMENT_COLORS.get(instrument, PALETTE["neutral"]),
            marker=marker,
            s=60,
            edgecolor="white",
            linewidth=0.5,
            label=label,
        )

    ax.axhline(0.0, color=PALETTE["neutral"], linewidth=0.6, alpha=0.6)
    ax.set_xlabel("Realised standard deviation (annualised return)")
    ax.set_ylabel("Mean annualised return")
    ax.set_title(title, loc="left", pad=14)
    ax.text(0.0, 1.04, subtitle, transform=ax.transAxes, fontsize=9, color=PALETTE["neutral"])
    ax.legend(loc="lower left", frameon=False, fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure #6 — Tranche fair-to-par vs ρ                                        #
# --------------------------------------------------------------------------- #


def fig_tranche_price_vs_rho(
    *,
    sweep_df: pd.DataFrame,
    title: str = "Tranche fair price vs correlation",
    subtitle: str = "Each curve is fair_to_par across a one-at-a-time sweep on rho",
) -> Figure:
    """Plot per-tranche fair_to_par against the copula correlation grid."""
    from .style import INSTRUMENT_COLORS

    apply_style()
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for inst, sub in sweep_df.groupby("instrument"):
        sub_sorted = sub.sort_values("value")
        ax.plot(
            sub_sorted["value"],
            sub_sorted["fair_to_par"],
            label=str(inst).replace("_", " ").title(),
            color=INSTRUMENT_COLORS.get(str(inst), PALETTE["neutral"]),
            linewidth=1.4,
            marker="o",
            markersize=4,
        )
    ax.axhline(1.0, color=PALETTE["neutral"], linewidth=0.7, linestyle="--", alpha=0.6)
    ax.set_xlabel(r"Copula correlation $\rho$")
    ax.set_ylabel("Fair price / par")
    ax.set_title(title, loc="left", pad=14)
    ax.text(0.0, 1.04, subtitle, transform=ax.transAxes, fontsize=9, color=PALETTE["neutral"])
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure #10 — Insurance break-even surface                                   #
# --------------------------------------------------------------------------- #


def fig_insurance_breakeven_surface(
    *,
    coverage_grid: np.ndarray,
    premium_grid: np.ndarray,
    npv_matrix: np.ndarray,
    title: str = "Insurance break-even surface — equity tranche",
    subtitle: str = "NPV with insurance minus uninsured baseline; the dashed contour is the zero frontier.",
) -> Figure:
    """Heatmap of (coverage, premium) → ΔNPV with the zero contour overlaid."""
    apply_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    pcm = ax.pcolormesh(coverage_grid, premium_grid, npv_matrix, cmap="RdBu", shading="auto")
    ax.contour(
        coverage_grid,
        premium_grid,
        npv_matrix,
        levels=[0.0],
        colors=PALETTE["neutral"],
        linewidths=1.0,
        linestyles="--",
    )
    cbar = fig.colorbar(pcm, ax=ax)
    cbar.set_label("ΔNPV vs uninsured (par fraction)")
    ax.set_xlabel("Coverage cap")
    ax.set_ylabel("Annual premium (% of par)")
    ax.set_title(title, loc="left", pad=14)
    ax.text(0.0, 1.04, subtitle, transform=ax.transAxes, fontsize=9, color=PALETTE["neutral"])
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure #11 — Tornado sensitivity                                            #
# --------------------------------------------------------------------------- #


def fig_tornado_sensitivity(
    *,
    base_value: float,
    deltas_by_param: dict[str, tuple[float, float]],
    title: str = "Tornado sensitivity — senior tranche fair_to_par",
    subtitle: str = "Each bar shows the change in fair_to_par when the parameter is shocked ±20 %.",
) -> Figure:
    """Horizontal bars sorted by absolute impact (largest at top)."""
    apply_style()
    items = sorted(
        deltas_by_param.items(),
        key=lambda kv: max(abs(kv[1][0]), abs(kv[1][1])),
        reverse=True,
    )
    labels = [k for k, _ in items]
    lows = [v[0] - base_value for _, v in items]
    highs = [v[1] - base_value for _, v in items]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.5 * len(labels) + 2)))
    ax.barh(y, lows, color=PALETTE["equity"], edgecolor="white", height=0.6, label="−20 %")
    ax.barh(y, highs, color=PALETTE["senior"], edgecolor="white", height=0.6, label="+20 %")
    ax.axvline(0.0, color=PALETTE["neutral"], linewidth=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Δ fair_to_par vs baseline")
    ax.set_title(title, loc="left", pad=14)
    ax.text(0.0, 1.04, subtitle, transform=ax.transAxes, fontsize=9, color=PALETTE["neutral"])
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure #12 — Stress heatmap                                                 #
# --------------------------------------------------------------------------- #


def fig_stress_heatmap(
    *,
    scenarios: list[str],
    instruments: list[str],
    values: np.ndarray,
    title: str = "Fair_to_par across stress scenarios",
    subtitle: str = "Rows = instruments, columns = scenarios; cells annotated with their value.",
    cmap: str = "RdYlGn",
) -> Figure:
    """Annotated heatmap with one cell per (instrument, scenario)."""
    apply_style()
    fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(scenarios) + 3), 0.5 * len(instruments) + 2))
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=0.4, vmax=1.2)
    ax.set_xticks(np.arange(len(scenarios)))
    ax.set_xticklabels([s.replace("_", " ") for s in scenarios])
    ax.set_yticks(np.arange(len(instruments)))
    ax.set_yticklabels([i.replace("_", " ").title() for i in instruments])
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(
                j, i, f"{values[i, j]:.2f}", ha="center", va="center", fontsize=8, color="black"
            )
    fig.colorbar(im, ax=ax, label="fair_to_par")
    ax.set_title(title, loc="left", pad=14)
    ax.text(0.0, 1.04, subtitle, transform=ax.transAxes, fontsize=9, color=PALETTE["neutral"])
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Figure #14 / #15 — Bootstrap CI fan chart + headline summary                #
# --------------------------------------------------------------------------- #


def fig_fairprice_fanchart(
    *,
    results: pd.DataFrame,
    insurance: str = "none",
    title: str = "Bootstrap 95 % confidence intervals on fair price",
    subtitle: str = "Whiskers are 1 000-resample percentile bootstrap of the MC paths.",
) -> Figure:
    """Per-instrument bar chart of fair_to_par with 95 % bootstrap whiskers."""
    from .style import CREDIT_MODEL_COLORS

    apply_style()
    sub = results[results["insurance"] == insurance].copy()
    instruments = ["model_a", "model_b", "equity", "mezzanine", "senior"]
    sub = sub[sub["instrument"].isin(instruments)]
    models = sorted(sub["credit_model"].unique())
    width = 0.25
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    x = np.arange(len(instruments))
    for i, model in enumerate(models):
        m_sub = sub[sub["credit_model"] == model].set_index("instrument").reindex(instruments)
        centres = (m_sub["fair_price"] / m_sub["initial_price"]).values
        lows = (m_sub["fair_price_lo"] / m_sub["initial_price"]).values
        highs = (m_sub["fair_price_hi"] / m_sub["initial_price"]).values
        err = np.vstack([centres - lows, highs - centres])
        offset = (i - (len(models) - 1) / 2) * width
        ax.bar(
            x + offset,
            centres,
            width=width,
            yerr=err,
            color=CREDIT_MODEL_COLORS.get(model, PALETTE["neutral"]),
            edgecolor="white",
            capsize=3,
            label=model.replace("_", " ").title(),
        )
    ax.axhline(1.0, color=PALETTE["neutral"], linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([i.replace("_", " ").title() for i in instruments])
    ax.set_ylabel("Fair price / par")
    ax.set_title(title, loc="left", pad=14)
    ax.text(0.0, 1.04, subtitle, transform=ax.transAxes, fontsize=9, color=PALETTE["neutral"])
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


def fig_headline_summary(
    *,
    results: pd.DataFrame,
    title: str = "Headline summary — fair price per instrument and credit model",
    subtitle: str = "Without insurance. Whiskers are bootstrap 95 % CIs.",
) -> Figure:
    """Single cover figure (no-insurance), suitable as the report headline."""
    return fig_fairprice_fanchart(results=results, insurance="none", title=title, subtitle=subtitle)


def fig_insurance_comparison(
    *,
    results: pd.DataFrame,
    credit_model: str = "gaussian_copula",
    title: str = "Insurance comparison — actuarial vs option-theoretic",
    subtitle: str = "Fair price / par per instrument under no insurance, actuarial pricing, and option-theoretic pricing.",
) -> Figure:
    """Grouped bars of fair_to_par across the three insurance regimes."""
    apply_style()
    sub = results[results["credit_model"] == credit_model]
    instruments = ["model_a", "model_b", "equity", "mezzanine", "senior"]
    regimes = ["none", "actuarial", "option_theoretic"]
    regime_label = {
        "none": "No insurance",
        "actuarial": "Actuarial",
        "option_theoretic": "Option-theor.",
    }
    regime_palette = {
        "none": PALETTE["neutral"],
        "actuarial": PALETTE["mezzanine"],
        "option_theoretic": PALETTE["senior"],
    }
    fig, ax = plt.subplots(figsize=(9.0, 4.6))
    x = np.arange(len(instruments))
    width = 0.27
    for i, regime in enumerate(regimes):
        r = sub[sub["insurance"] == regime].set_index("instrument").reindex(instruments)
        values = (r["fair_price"] / r["initial_price"]).values
        offset = (i - 1) * width
        ax.bar(
            x + offset,
            values,
            width=width,
            color=regime_palette[regime],
            edgecolor="white",
            label=regime_label[regime],
        )
    ax.axhline(1.0, color=PALETTE["neutral"], linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([i.replace("_", " ").title() for i in instruments])
    ax.set_ylabel("Fair price / par")
    ax.set_title(title, loc="left", pad=14)
    ax.text(0.0, 1.04, subtitle, transform=ax.transAxes, fontsize=9, color=PALETTE["neutral"])
    ax.legend(loc="best", frameon=False, fontsize=9)
    fig.tight_layout()
    return fig


__all__ = [
    "FRENCH_RECESSIONS",
    "fig_calibration_diagnostics",
    "fig_fairprice_fanchart",
    "fig_headline_summary",
    "fig_insurance_breakeven_surface",
    "fig_insurance_comparison",
    "fig_loss_distributions",
    "fig_mc_convergence",
    "fig_pareto_frontier",
    "fig_paris_price_index",
    "fig_riskreturn_bars",
    "fig_stress_heatmap",
    "fig_tail_dependence",
    "fig_tornado_sensitivity",
    "fig_tranche_price_vs_rho",
    "fig_waterfall_explainer",
]
