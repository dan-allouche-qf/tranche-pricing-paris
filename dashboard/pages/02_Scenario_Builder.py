"""Live scenario builder — sliders feed a small-N Monte Carlo and refresh the view."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import streamlit as st

from tranche_pricing.credit.lgd import BetaLGDParams
from tranche_pricing.markets.price_gbm import GBMParams
from tranche_pricing.markets.rates_vasicek import VasicekParams
from tranche_pricing.pricing import compare_credit_models
from tranche_pricing.simulation import SimulationConfig
from tranche_pricing.viz import figures
from tranche_pricing.viz.style import apply_style
from tranche_pricing.waterfall.tranches import Tranche

apply_style()
st.title("Scenario builder")
st.caption(
    "Adjust the headline parameters and re-run the Monte Carlo under the Gaussian "
    "one-factor copula. We cache the simulation result on the parameter tuple, so "
    "moving a single slider only re-runs when needed."
)

with st.sidebar:
    st.header("Parameters")
    par = st.slider("Building par (M€)", 1.0, 50.0, 31.5, 0.5)
    n_obligors = st.slider("Number of apartments", 8, 100, 70, 1)
    horizon = st.slider("Horizon (years)", 3, 20, 10, 1)
    pd_annual = st.slider("Annual default probability", 0.005, 0.10, 0.03, 0.005)
    rho = st.slider("Copula correlation ρ", 0.0, 0.6, 0.15, 0.05)
    gbm_mu = st.slider("GBM drift μ", -0.05, 0.10, 0.025, 0.005)
    gbm_sigma = st.slider("GBM volatility σ", 0.01, 0.15, 0.05, 0.005)
    n_paths = st.select_slider(
        "Monte Carlo paths", options=[500, 1000, 2000, 5000, 10000], value=2000
    )


@st.cache_data(show_spinner=False)
def run(
    par_m_eur: float,
    n_apt: int,
    h: int,
    pd_a: float,
    rho_val: float,
    mu: float,
    sigma: float,
    paths: int,
):
    """Cached single-model comparison; returns a pandas DataFrame."""
    sim = SimulationConfig(
        n_obligors=n_apt,
        horizon_years=float(h),
        steps_per_year=12,
        par=par_m_eur * 1e6,
        gross_yield=0.035,
        maintenance_pct=0.015,
        initial_rate=0.03,
        pd_terminal=min(0.95, 1.0 - (1.0 - pd_a) ** h),
        credit_model="gaussian_copula",
        rho=rho_val,
        n_paths=paths,
        master_seed=20260519,
    )
    df, _ = compare_credit_models(
        sim,
        gbm=GBMParams(mu=mu, sigma=sigma),
        vasicek=VasicekParams(a=0.20, b=0.025, sigma_r=0.01),
        lgd=BetaLGDParams(mean=0.85, std=0.12),
        tranches=[
            Tranche("equity", 0.0, 0.25),
            Tranche("mezzanine", 0.25, 0.60),
            Tranche("senior", 0.60, 1.0),
        ],
        coupons={"senior": 0.03, "mezzanine": 0.06, "equity": 0.0},
        models=("gaussian_copula",),
    )
    return df


with st.spinner("Running Monte Carlo …"):
    df = run(par, n_obligors, horizon, pd_annual, rho, gbm_mu, gbm_sigma, n_paths)

st.subheader("Per-instrument fair prices and risk metrics")
display_cols = [
    "instrument",
    "initial_price",
    "fair_price",
    "fair_to_par",
    "mean_ann_return",
    "risk_std",
    "risk_sharpe",
    "risk_var_95",
    "prob_negative_return",
]
st.dataframe(
    df[display_cols]
    .set_index("instrument")
    .style.format(
        {
            "initial_price": "{:,.0f}",
            "fair_price": "{:,.0f}",
            "fair_to_par": "{:.3f}",
            "mean_ann_return": "{:.2%}",
            "risk_std": "{:.2%}",
            "risk_sharpe": "{:.2f}",
            "risk_var_95": "{:.2%}",
            "prob_negative_return": "{:.1%}",
        }
    )
)

st.subheader("Risk-return frontier (live)")
df_for_fig = df.assign(insurance="none")
fig = figures.fig_pareto_frontier(results=df_for_fig)
st.pyplot(fig)
