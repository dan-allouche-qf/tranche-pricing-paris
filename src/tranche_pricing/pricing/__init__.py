"""Per-instrument pricing and the joint-model comparison runner."""

from __future__ import annotations

from . import instruments, model_compare, sensitivity, tranche_pricer
from .instruments import InstrumentCashFlows, extract_all
from .model_compare import compare_credit_models
from .tranche_pricer import (
    InstrumentPricing,
    bootstrap_fair_price_ci,
    price_all,
    price_instrument,
    solve_fair_coupon,
    solve_fair_coupons_for_all,
)

__all__ = [
    "InstrumentCashFlows",
    "InstrumentPricing",
    "bootstrap_fair_price_ci",
    "compare_credit_models",
    "extract_all",
    "instruments",
    "model_compare",
    "price_all",
    "price_instrument",
    "sensitivity",
    "solve_fair_coupon",
    "solve_fair_coupons_for_all",
    "tranche_pricer",
]
