"""Parametric sensitivity sweeps used by figures 6 and 11.

Each ``sweep_*`` helper takes a baseline ``SimulationConfig`` plus a grid of
values for one input and returns a tidy ``pandas.DataFrame`` of per-tranche
``fair_to_par`` at every grid point. The simulations are run under the
Gaussian one-factor copula (the baseline of the working paper) — multi-model
sweeps would be straightforward to add but blow up the runtime.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import pandas as pd

from ..credit.lgd import BetaLGDParams
from ..markets.price_gbm import GBMParams
from ..markets.rates_vasicek import VasicekParams
from ..simulation import SimulationConfig
from .model_compare import compare_credit_models
from .tranche_pricer import price_all  # noqa: F401  (kept for the reader)


def _run_one(
    sim: SimulationConfig,
    *,
    gbm: GBMParams,
    vasicek: VasicekParams,
    lgd: BetaLGDParams,
    tranches,
    coupons,
) -> pd.DataFrame:
    df, _ = compare_credit_models(
        sim,
        gbm=gbm,
        vasicek=vasicek,
        lgd=lgd,
        tranches=tranches,
        coupons=coupons,
        models=("gaussian_copula",),
    )
    return df


def sweep_rho(
    base: SimulationConfig,
    *,
    rho_grid: Sequence[float],
    gbm: GBMParams,
    vasicek: VasicekParams,
    lgd: BetaLGDParams,
    tranches,
    coupons,
) -> pd.DataFrame:
    """Sweep the copula correlation ρ and return a tidy DataFrame."""
    rows = []
    for rho in rho_grid:
        sim = replace(base, rho=float(rho))
        df = _run_one(sim, gbm=gbm, vasicek=vasicek, lgd=lgd, tranches=tranches, coupons=coupons)
        for _, r in df.iterrows():
            rows.append(
                {
                    "param": "rho",
                    "value": float(rho),
                    "instrument": r["instrument"],
                    "fair_to_par": float(r["fair_to_par"]),
                    "mean_ann_return": float(r["mean_ann_return"]),
                    "risk_std": float(r["risk_std"]),
                }
            )
    return pd.DataFrame(rows)


def sweep_pd_annual(
    base: SimulationConfig,
    *,
    pd_grid: Sequence[float],
    horizon: float,
    gbm: GBMParams,
    vasicek: VasicekParams,
    lgd: BetaLGDParams,
    tranches,
    coupons,
) -> pd.DataFrame:
    """Sweep the annual default probability and return a tidy DataFrame."""
    rows = []
    for p in pd_grid:
        pd_t = min(0.95, 1.0 - (1.0 - p) ** horizon)
        sim = replace(base, pd_terminal=pd_t)
        df = _run_one(sim, gbm=gbm, vasicek=vasicek, lgd=lgd, tranches=tranches, coupons=coupons)
        for _, r in df.iterrows():
            rows.append(
                {
                    "param": "pd_annual",
                    "value": float(p),
                    "instrument": r["instrument"],
                    "fair_to_par": float(r["fair_to_par"]),
                    "mean_ann_return": float(r["mean_ann_return"]),
                    "risk_std": float(r["risk_std"]),
                }
            )
    return pd.DataFrame(rows)


def sweep_param(
    base: SimulationConfig,
    *,
    param: str,
    grid: Sequence[float],
    gbm: GBMParams,
    vasicek: VasicekParams,
    lgd: BetaLGDParams,
    tranches,
    coupons,
) -> pd.DataFrame:
    """Generic single-parameter OAT (one-at-a-time) sweep over a numeric attribute."""
    if not hasattr(base, param):
        raise ValueError(f"SimulationConfig has no attribute {param!r}.")
    rows = []
    for v in grid:
        # The single-attribute override is dynamic; the static type is checked
        # at sweep entry by ``hasattr(base, param)`` above.
        sim = replace(base, **{param: float(v)})  # type: ignore[arg-type]
        df = _run_one(sim, gbm=gbm, vasicek=vasicek, lgd=lgd, tranches=tranches, coupons=coupons)
        for _, r in df.iterrows():
            rows.append(
                {
                    "param": param,
                    "value": float(v),
                    "instrument": r["instrument"],
                    "fair_to_par": float(r["fair_to_par"]),
                    "mean_ann_return": float(r["mean_ann_return"]),
                    "risk_std": float(r["risk_std"]),
                }
            )
    return pd.DataFrame(rows)


__all__ = ["sweep_param", "sweep_pd_annual", "sweep_rho"]
