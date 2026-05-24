"""Vectorised risk metrics on return / cash-flow arrays.

Every function in this module accepts a 1-D numpy array of per-path scalar
values (typically a per-path realised return, a per-path PV, or a per-path
total cash flow) and returns a single scalar metric. The inputs do not have
to be sorted; the function handles that internally.

References for the conventions: Acerbi-Tasche (2002) for ES, Sortino-van der
Meer (1991) for the Sortino ratio, Magdon-Ismail-Atiya for Calmar, Keating-
Shadwick (2002) for Omega.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def value_at_risk(returns: NDArray[np.float64], alpha: float = 0.95) -> float:
    """Empirical Value-at-Risk at confidence ``alpha``.

    Sign convention: VaR is reported as a POSITIVE loss number (i.e. minus the
    lower-tail quantile of ``returns``). A higher VaR means more downside.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")
    arr = np.asarray(returns, dtype=float)
    return float(-np.quantile(arr, 1.0 - alpha))


def expected_shortfall(returns: NDArray[np.float64], alpha: float = 0.95) -> float:
    """Expected Shortfall (Acerbi-Tasche) at confidence ``alpha``.

    Reported as a positive loss number, same convention as :func:`value_at_risk`.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1).")
    arr = np.asarray(returns, dtype=float)
    threshold = np.quantile(arr, 1.0 - alpha)
    tail = arr[arr <= threshold]
    if tail.size == 0:
        return float(-threshold)
    return float(-tail.mean())


def sharpe_ratio(returns: NDArray[np.float64], rf: float = 0.0) -> float:
    """Sharpe ratio on the supplied returns.

    Computes ``(mean(returns) - rf) / std(returns, ddof=1)``. The caller is
    responsible for ensuring ``returns`` are on the desired annualisation
    basis; the pricing layer passes geometric-annualised total returns, so
    the resulting Sharpe is annualised-equivalent.
    """
    arr = np.asarray(returns, dtype=float)
    excess = arr - rf
    sd = float(excess.std(ddof=1))
    if sd == 0:
        return float("nan")
    return float(excess.mean() / sd)


def sortino_ratio(returns: NDArray[np.float64], target: float = 0.0) -> float:
    """Sortino ratio: excess mean over downside deviation around ``target``.

    Downside deviation uses the full-sample expectation
    ``sqrt(E[(target - r)_+**2])`` (Sortino-van der Meer 1991), matching the
    cheat-sheet formula in ``14_appendix_cheatsheet.tex``.
    """
    arr = np.asarray(returns, dtype=float)
    excess = arr - target
    downside_sq = np.where(excess < 0, excess**2, 0.0)
    downside_std = float(np.sqrt(downside_sq.mean()))
    if downside_std == 0:
        return float("inf") if excess.mean() > 0 else float("nan")
    return float(excess.mean() / downside_std)


def max_drawdown(cumulative_cashflows: NDArray[np.float64]) -> float:
    """Maximum drawdown on a 1-D cumulative cash-flow series.

    Returned as a positive fraction. ``0.20`` means a peak-to-trough loss of
    20% of the prior cumulative high.
    """
    arr = np.asarray(cumulative_cashflows, dtype=float)
    if arr.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(arr)
    drawdowns = (running_max - arr) / np.where(running_max != 0, running_max, 1.0)
    return float(drawdowns.max())


def calmar_ratio(
    returns: NDArray[np.float64],
    cumulative_cashflows: NDArray[np.float64],
) -> float:
    """Calmar ratio = annualised return divided by maximum drawdown."""
    mdd = max_drawdown(cumulative_cashflows)
    if mdd == 0:
        return float("nan")
    return float(np.mean(returns) / mdd)


def omega_ratio(returns: NDArray[np.float64], threshold: float = 0.0) -> float:
    """Omega ratio at the given threshold.

    Omega = E[(X - threshold)+] / E[(threshold - X)+]; values above 1 mean
    upside potential exceeds downside risk at the given threshold.
    """
    arr = np.asarray(returns, dtype=float)
    above = np.maximum(arr - threshold, 0.0).mean()
    below = np.maximum(threshold - arr, 0.0).mean()
    if below == 0:
        return float("inf") if above > 0 else float("nan")
    return float(above / below)


def tracking_error(returns: NDArray[np.float64], benchmark: NDArray[np.float64]) -> float:
    """Empirical tracking error: std of (returns - benchmark)."""
    diff = np.asarray(returns, dtype=float) - np.asarray(benchmark, dtype=float)
    return float(diff.std(ddof=1))


def summary(
    *,
    returns: NDArray[np.float64],
    cumulative_cashflows: NDArray[np.float64] | None = None,
    rf: float = 0.0,
    var_alpha: tuple[float, ...] = (0.95, 0.99),
    es_alpha: tuple[float, ...] = (0.95, 0.99),
    omega_threshold: float = 0.0,
) -> dict[str, float]:
    """Compute a wide summary of risk metrics on ``returns``.

    Always-on metrics: mean, std, skew (population), Sharpe, Sortino, Omega.
    VaR / ES are computed at every alpha in ``var_alpha`` / ``es_alpha``.
    """
    arr = np.asarray(returns, dtype=float)
    out: dict[str, float] = {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)),
        "skew": _sample_skew(arr),
        "sharpe": sharpe_ratio(arr, rf=rf),
        "sortino": sortino_ratio(arr, target=rf),
        "omega": omega_ratio(arr, threshold=omega_threshold),
    }
    for a in var_alpha:
        out[f"var_{int(a * 100)}"] = value_at_risk(arr, alpha=a)
    for a in es_alpha:
        out[f"es_{int(a * 100)}"] = expected_shortfall(arr, alpha=a)
    if cumulative_cashflows is not None:
        out["max_drawdown"] = max_drawdown(cumulative_cashflows)
        out["calmar"] = calmar_ratio(arr, cumulative_cashflows)
    return out


def _sample_skew(arr: NDArray[np.float64]) -> float:
    """Population skewness (matches scipy.stats.skew with bias=True)."""
    if arr.size < 3:
        return float("nan")
    centered = arr - arr.mean()
    m2 = float((centered**2).mean())
    m3 = float((centered**3).mean())
    if m2 == 0:
        return float("nan")
    return float(m3 / m2**1.5)


__all__ = [
    "calmar_ratio",
    "expected_shortfall",
    "max_drawdown",
    "omega_ratio",
    "sharpe_ratio",
    "sortino_ratio",
    "summary",
    "tracking_error",
    "value_at_risk",
]
