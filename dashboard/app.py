"""Streamlit entry point — Overview page.

Run with::

    streamlit run dashboard/app.py

The multi-page app pulls every page from ``dashboard/pages/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from tranche_pricing.viz.style import apply_style, register_plotly_template

st.set_page_config(
    page_title="Tranche pricing on Paris rent",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_style()
register_plotly_template()

st.title("Selling Property Rental Ownership")
st.caption(
    "A multi-model tranche-pricing study of Paris residential rental cash flows · "
    "Dan Allouche · May 2026"
)

st.markdown(
    """
This interactive companion to the working paper lets you re-run the Monte
Carlo pipeline under arbitrary scenarios, compare the three credit models
side-by-side, and reproduce the stress / backtest exercises.

Use the sidebar to navigate:

| Page | What it does |
|---|---|
| **Overview** | This landing page. |
| **Scenario builder** | Live sliders for the headline parameters (correlation, default rate, drift / vol). |
| **Tranche pricer** | Pick a credit model and inspect the fair prices and risk metrics. |
| **Stress lab** | Apply the GFC / COVID / 2022 rate-hike overlays. |
| **Backtest replay** | Animate the 2008 / 2020 episodes through the cumulative-loss path. |
"""
)

st.subheader("Headline figure")
fig_path = ROOT / "artifacts/figures/fig_paris_price_index.png"
if fig_path.exists():
    st.image(str(fig_path), use_column_width=True)
else:
    st.info("No figure yet — run `make data && make figures` to populate `artifacts/figures/`.")

st.subheader("Latest results")
results_path = ROOT / "artifacts/results.csv"
if results_path.exists():
    import pandas as pd

    df = pd.read_csv(results_path)
    gauss = df[(df["credit_model"] == "gaussian_copula") & (df["insurance"] == "none")]
    st.dataframe(
        gauss[
            [
                "instrument",
                "initial_price",
                "fair_price",
                "fair_to_par",
                "mean_ann_return",
                "risk_sharpe",
                "risk_var_95",
                "prob_negative_return",
            ]
        ]
        .set_index("instrument")
        .style.format(
            {
                "initial_price": "{:,.0f}",
                "fair_price": "{:,.0f}",
                "fair_to_par": "{:.3f}",
                "mean_ann_return": "{:.2%}",
                "risk_sharpe": "{:.2f}",
                "risk_var_95": "{:.2%}",
                "prob_negative_return": "{:.1%}",
            }
        )
    )
else:
    st.info("No `artifacts/results.csv` yet — run `make mc` first.")

st.markdown(
    """
---
**Codebase:** [`tranche-pricing-paris`](https://github.com/dan-allouche/tranche-pricing-paris) ·
**Working paper:** `report/main.pdf`.
"""
)
