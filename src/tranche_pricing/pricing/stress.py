"""Historical stress replays via the ``stress_*.yaml`` overlays.

Each scenario applies a documented perturbation to the price drift / vol,
the short-rate level and the marginal default probability. We rerun the
joint Gaussian / Student-t / Cox simulation under each overlay and write
the per-instrument fair-to-par and tail metrics to
``artifacts/stress_results.csv`` so the working paper's stress section can
read the numbers directly.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pandas as pd

from ..config import Config, load_config
from ..credit.lgd import BetaLGDParams
from ..markets.price_gbm import GBMParams
from ..markets.rates_vasicek import VasicekParams
from ..simulation import SimulationConfig
from ..waterfall.tranches import Tranche
from .model_compare import compare_credit_models

# YAML overlays shipped under config/.
DEFAULT_SCENARIOS: tuple[str, ...] = (
    "stress_gfc.yaml",
    "stress_covid.yaml",
    "stress_rates2022.yaml",
)


def _apply_overlay(
    base_cfg: Config,
    overlay_cfg: Config,
    *,
    base_sim: SimulationConfig,
    base_gbm: GBMParams,
    base_vas: VasicekParams,
) -> tuple[SimulationConfig, GBMParams, VasicekParams]:
    """Apply a stress overlay to the baseline sim / market parameters."""
    overlay = overlay_cfg.overlay
    cur_gbm = base_gbm
    cur_vas = base_vas
    sim = base_sim
    if overlay is not None and overlay.price_dynamics is not None:
        cur_gbm = GBMParams(
            mu=base_gbm.mu + overlay.price_dynamics.mu_shift_pct,
            sigma=base_gbm.sigma * overlay.price_dynamics.sigma_multiplier,
        )
    if overlay is not None and overlay.rates is not None:
        cur_vas = VasicekParams(
            a=base_vas.a,
            b=base_vas.b + overlay.rates.delta_bps / 10000.0,
            sigma_r=base_vas.sigma_r,
        )
    if overlay is not None and overlay.credit is not None:
        pd_a = base_cfg.credit.pd_annual_init * overlay.credit.pd_multiplier
        pd_t = min(0.95, 1.0 - (1.0 - pd_a) ** base_sim.horizon_years)
        sim = replace(sim, pd_terminal=pd_t)
    return sim, cur_gbm, cur_vas


def run_stress_replays(
    base_cfg: Config,
    *,
    base_sim: SimulationConfig,
    gbm: GBMParams,
    vasicek: VasicekParams,
    lgd: BetaLGDParams,
    tranches: list[Tranche],
    coupons: dict[str, float],
    scenarios: Sequence[str] = DEFAULT_SCENARIOS,
    config_dir: Path | None = None,
) -> pd.DataFrame:
    """Run every stress overlay and assemble a tidy results table.

    Returns
    -------
    pandas.DataFrame
        Columns ``scenario, credit_model, instrument, fair_to_par,
        mean_ann_return, risk_std, risk_var_95, risk_es_95,
        prob_negative_return`` — one row per (scenario, model, instrument).
    """
    cfg_dir = config_dir if config_dir is not None else Path("config")
    rows: list[dict[str, object]] = []

    # Baseline (no overlay) for reference.
    df_base, _ = compare_credit_models(
        base_sim, gbm=gbm, vasicek=vasicek, lgd=lgd, tranches=tranches, coupons=coupons
    )
    for _, r in df_base.iterrows():
        rows.append(
            {
                "scenario": "baseline",
                "credit_model": r["credit_model"],
                "instrument": r["instrument"],
                "fair_to_par": float(r["fair_to_par"]),
                "mean_ann_return": float(r["mean_ann_return"]),
                "risk_std": float(r["risk_std"]),
                "risk_var_95": float(r["risk_var_95"]),
                "risk_es_95": float(r["risk_es_95"]),
                "prob_negative_return": float(r["prob_negative_return"]),
            }
        )

    for scenario_file in scenarios:
        overlay_cfg = load_config(cfg_dir / scenario_file)
        sim_s, gbm_s, vas_s = _apply_overlay(
            base_cfg, overlay_cfg, base_sim=base_sim, base_gbm=gbm, base_vas=vasicek
        )
        df_s, _ = compare_credit_models(
            sim_s, gbm=gbm_s, vasicek=vas_s, lgd=lgd, tranches=tranches, coupons=coupons
        )
        for _, r in df_s.iterrows():
            rows.append(
                {
                    "scenario": overlay_cfg.scenario.name,
                    "credit_model": r["credit_model"],
                    "instrument": r["instrument"],
                    "fair_to_par": float(r["fair_to_par"]),
                    "mean_ann_return": float(r["mean_ann_return"]),
                    "risk_std": float(r["risk_std"]),
                    "risk_var_95": float(r["risk_var_95"]),
                    "risk_es_95": float(r["risk_es_95"]),
                    "prob_negative_return": float(r["prob_negative_return"]),
                }
            )

    return pd.DataFrame(rows)


__all__ = ["DEFAULT_SCENARIOS", "run_stress_replays"]
