"""Tranche pricer page — compare the three credit models side-by-side."""

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
st.title("Tranche pricer — three credit models")
st.caption(
    "Same scenario, three credit models. The Student-t copula widens the senior "
    "loss tail; the Cox doubly stochastic model produces a tighter mass around "
    "the mean."
)

with st.sidebar:
    st.header("Scenario")
    par = st.slider("Par (M€)", 1.0, 50.0, 31.5, 0.5)
    n_obligors = st.slider("Apartments", 20, 120, 70, 10)
    rho = st.slider("ρ", 0.0, 0.6, 0.15, 0.05)
    nu = st.slider("Student-t ν", 3.0, 30.0, 5.0, 1.0)
    pd_annual = st.slider("Annual default probability", 0.005, 0.10, 0.03, 0.005)
    n_paths = st.select_slider("Monte Carlo paths", options=[500, 1000, 2000, 5000], value=2000)


@st.cache_data(show_spinner=False)
def run(par_m: float, n_apt: int, rho_v: float, nu_v: float, pd_a: float, paths: int):
    sim = SimulationConfig(
        n_obligors=n_apt,
        horizon_years=10.0,
        steps_per_year=12,
        par=par_m * 1e6,
        gross_yield=0.035,
        maintenance_pct=0.015,
        initial_rate=0.03,
        pd_terminal=min(0.95, 1.0 - (1.0 - pd_a) ** 10),
        credit_model="gaussian_copula",
        rho=rho_v,
        nu=nu_v,
        n_paths=paths,
        master_seed=20260519,
    )
    df, outputs = compare_credit_models(
        sim,
        gbm=GBMParams(mu=0.025, sigma=0.05),
        vasicek=VasicekParams(a=0.20, b=0.025, sigma_r=0.01),
        lgd=BetaLGDParams(mean=0.85, std=0.12),
        tranches=[
            Tranche("equity", 0.0, 0.25),
            Tranche("mezzanine", 0.25, 0.60),
            Tranche("senior", 0.60, 1.0),
        ],
        coupons={"senior": 0.03, "mezzanine": 0.06, "equity": 0.0},
    )
    losses = {name: out.cumulative_loss[:, -1] for name, out in outputs.items()}
    return df, losses


with st.spinner("Running comparison …"):
    df, losses = run(par, n_obligors, rho, nu, pd_annual, n_paths)

st.subheader("Fair-to-par by credit model")
pivot = df.pivot(index="instrument", columns="credit_model", values="fair_to_par").reindex(
    ["model_a", "model_b", "equity", "mezzanine", "senior"]
)
st.dataframe(pivot.style.format("{:.3f}"))

st.subheader("Cumulative loss distribution")
fig = figures.fig_loss_distributions(losses_by_model=losses)
st.pyplot(fig)

st.subheader("Lower tail dependence under the Student-t copula")
fig2 = figures.fig_tail_dependence()
st.pyplot(fig2)
