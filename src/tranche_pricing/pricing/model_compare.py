"""Joint pricing across the three credit models for a single calibration.

The function :func:`compare_credit_models` runs the same Monte Carlo
simulation three times — once under each credit model — and returns a
flat ``pandas.DataFrame`` of per-instrument metrics that the working paper
can drop straight into a table.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import pandas as pd

from ..credit.lgd import BetaLGDParams
from ..markets.price_gbm import GBMParams
from ..markets.rates_vasicek import VasicekParams
from ..simulation import SimulationConfig, run_simulation
from ..simulation.engine import CreditModelName, SimulationOutput
from ..waterfall.tranches import Tranche
from .instruments import extract_all
from .tranche_pricer import price_all

DEFAULT_MODELS: tuple[CreditModelName, ...] = (
    "gaussian_copula",
    "student_t_copula",
    "cox_intensity",
)


def compare_credit_models(
    base_cfg: SimulationConfig,
    *,
    gbm: GBMParams,
    vasicek: VasicekParams,
    lgd: BetaLGDParams,
    tranches: list[Tranche],
    coupons: dict[str, float],
    models: Sequence[CreditModelName] = DEFAULT_MODELS,
    bootstrap_resamples: int = 0,
) -> tuple[pd.DataFrame, dict[str, SimulationOutput]]:
    """Run the simulation under each credit model and assemble a tidy table.

    Returns
    -------
    (df, outputs)
        ``df`` is a DataFrame with one row per (model, instrument); ``outputs``
        is a dict mapping model name to the raw ``SimulationOutput`` for
        downstream diagnostics (e.g. variance-reduction studies).
    """
    rows: list[dict[str, Any]] = []
    outputs: dict[str, SimulationOutput] = {}
    for model in models:
        cfg = replace(base_cfg, credit_model=model)
        out = run_simulation(
            cfg,
            gbm=gbm,
            vasicek=vasicek,
            lgd=lgd,
            tranches=tranches,
            coupons=coupons,
        )
        outputs[model] = out
        instruments = extract_all(out)
        pricings = price_all(instruments, out=out, bootstrap_resamples=bootstrap_resamples)
        for inst_name, pricing in pricings.items():
            record = pricing.as_record()
            record["credit_model"] = model
            record["instrument"] = inst_name
            rows.append(record)

    df = pd.DataFrame(rows)
    # Re-order columns: identifiers first, then summary numbers, then risk.
    leading = ["credit_model", "instrument", "initial_price", "fair_price", "fair_to_par"]
    risk_cols = [c for c in df.columns if c.startswith("risk_")]
    return_cols = [
        c
        for c in df.columns
        if (c.endswith("return") or c.startswith("prob_")) and c not in risk_cols
    ]
    seen = set(leading) | set(risk_cols) | set(return_cols)
    other = [c for c in df.columns if c not in seen]
    return df[leading + return_cols + risk_cols + other], outputs


__all__ = ["DEFAULT_MODELS", "compare_credit_models"]
