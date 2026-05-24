"""Backtest replay — animate the cumulative loss path under a chosen scenario."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd
import streamlit as st

from tranche_pricing.config import load_config
from tranche_pricing.pricing import runner as pricing_runner
from tranche_pricing.simulation import run_simulation
from tranche_pricing.viz.style import PALETTE, apply_style

apply_style()
st.title("Backtest replay")
st.caption(
    "Pick a historical episode and pick a Monte Carlo path index; the chart shows the "
    "cumulative aggregate loss accumulated period by period, alongside the live "
    "tranche notionals."
)

baseline_cfg = load_config(ROOT / "config/paris_intermediate.yaml")
sim_base, gbm, vas, lgd, tranches, coupons = pricing_runner.build_inputs_from_yaml(baseline_cfg)

with st.sidebar:
    scenario = st.selectbox(
        "Episode",
        ["baseline", "stress_gfc", "stress_covid", "stress_rates2022"],
        index=1,
    )
    n_paths = st.select_slider("Monte Carlo sample size", options=[200, 500, 1000, 2000], value=500)
    path_idx = st.slider("Path index", 0, n_paths - 1, 0, 1)


@st.cache_data(show_spinner=False)
def run(scenario_name: str, paths: int):
    cur_gbm = gbm
    cur_vas = vas
    sim = replace(sim_base, n_paths=paths)
    if scenario_name != "baseline":
        from tranche_pricing.markets.price_gbm import GBMParams
        from tranche_pricing.markets.rates_vasicek import VasicekParams

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
    out = run_simulation(
        sim, gbm=cur_gbm, vasicek=cur_vas, lgd=lgd, tranches=tranches, coupons=coupons
    )
    return out


with st.spinner(f"Simulating '{scenario}' …"):
    out = run(scenario, n_paths)

dt = float(out.dt)
n_steps = out.n_steps
t_grid = np.arange(n_steps + 1) * dt

import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(
    t_grid,
    out.cumulative_loss[path_idx],
    color=PALETTE["equity"],
    linewidth=1.6,
    label="Cumulative loss / par",
)
ax.set_ylim(0, max(0.05, out.cumulative_loss[path_idx, -1] * 1.4))
ax.set_xlabel("Years from inception")
ax.set_ylabel("Loss fraction of par")
ax.axhline(0.25, color=PALETTE["mezzanine"], linestyle="--", linewidth=0.9, alpha=0.7)
ax.text(
    t_grid[-1], 0.255, "equity exhausted (0.25)", color=PALETTE["mezzanine"], fontsize=8, ha="right"
)
ax.axhline(0.60, color=PALETTE["senior"], linestyle="--", linewidth=0.9, alpha=0.7)
ax.text(
    t_grid[-1], 0.61, "mezzanine exhausted (0.60)", color=PALETTE["senior"], fontsize=8, ha="right"
)
ax.legend(loc="upper left", frameon=False)
st.pyplot(fig)

st.subheader("Tranche notionals along the chosen path")
notional_df = pd.DataFrame(
    {
        "year": t_grid,
        **{
            name: out.waterfall.notional_path[name][path_idx] / out.config.par
            for name in ["equity", "mezzanine", "senior"]
        },
    }
)
st.line_chart(notional_df.set_index("year"))
