"""Stress overlays — replay the GFC / COVID / 2022 rate-hike cycles."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd
import streamlit as st

from tranche_pricing.config import load_config
from tranche_pricing.markets.price_gbm import GBMParams
from tranche_pricing.markets.rates_vasicek import VasicekParams
from tranche_pricing.pricing import compare_credit_models
from tranche_pricing.pricing import runner as pricing_runner
from tranche_pricing.viz.style import apply_style

apply_style()
st.title("Stress lab")
st.caption(
    "Pick a historical episode from the dropdown; the overlay tweaks the price drift, "
    "the default rate and the short rate as documented in `config/stress_*.yaml`."
)

baseline_cfg = load_config(ROOT / "config/paris_intermediate.yaml")
sim_base, gbm, vas, lgd, tranches, coupons = pricing_runner.build_inputs_from_yaml(baseline_cfg)

scenario = st.selectbox(
    "Stress scenario",
    ["baseline", "stress_gfc", "stress_covid", "stress_rates2022"],
    index=0,
)


@st.cache_data(show_spinner=False)
def run(scenario_name: str, n_paths: int = 2000) -> pd.DataFrame:
    cur_gbm = gbm
    cur_vas = vas
    sim = replace(sim_base, n_paths=n_paths)
    if scenario_name != "baseline":
        stress = load_config(ROOT / f"config/{scenario_name}.yaml")
        overlay = stress.overlay
        if overlay and overlay.price_dynamics:
            cur_gbm = GBMParams(
                mu=gbm.mu + overlay.price_dynamics.mu_shift_pct,
                sigma=gbm.sigma * overlay.price_dynamics.sigma_multiplier,
            )
        if overlay and overlay.rates:
            cur_vas = VasicekParams(
                a=vas.a, b=vas.b + overlay.rates.delta_bps / 10000, sigma_r=vas.sigma_r
            )
        if overlay and overlay.credit:
            pd_a = baseline_cfg.credit.pd_annual_init * overlay.credit.pd_multiplier
            pd_t = min(0.95, 1.0 - (1.0 - pd_a) ** sim_base.horizon_years)
            sim = replace(sim, pd_terminal=pd_t)
    df, _ = compare_credit_models(
        sim,
        gbm=cur_gbm,
        vasicek=cur_vas,
        lgd=lgd,
        tranches=tranches,
        coupons=coupons,
        models=("gaussian_copula",),
    )
    return df


with st.spinner(f"Running '{scenario}' …"):
    df = run(scenario)

st.subheader("Per-instrument outcomes under the selected overlay")
display = df.set_index("instrument")[
    [
        "fair_to_par",
        "mean_ann_return",
        "risk_std",
        "risk_var_95",
        "risk_es_95",
        "prob_negative_return",
    ]
]
st.dataframe(
    display.style.format(
        {
            "fair_to_par": "{:.3f}",
            "mean_ann_return": "{:.2%}",
            "risk_std": "{:.2%}",
            "risk_var_95": "{:.2%}",
            "risk_es_95": "{:.2%}",
            "prob_negative_return": "{:.1%}",
        }
    )
)

st.subheader("Comparison vs baseline (Δ percentage points)")
if scenario != "baseline":
    base = run("baseline").set_index("instrument")[display.columns]
    delta = (display - base).style.format("{:+.4f}")
    st.dataframe(delta)
else:
    st.info("Selected scenario IS the baseline.")
