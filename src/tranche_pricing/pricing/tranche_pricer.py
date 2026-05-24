"""Pricing and risk-adjusted metrics per instrument.

Given the per-instrument cash flows extracted by :mod:`instruments`, this
module computes the fair price, the realised IRR per path, and a full risk
summary (Sharpe, Sortino, Calmar, Omega, VaR / ES at multiple confidence
levels) for the comparison table that feeds the working paper and the
Streamlit dashboard.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq, fsolve

from ..risk import metrics as risk_metrics
from ..simulation.engine import SimulationOutput
from .instruments import InstrumentCashFlows


@dataclass(slots=True)
class InstrumentPricing:
    """Per-instrument pricing and risk results."""

    name: str
    initial_price: float
    total_pv_per_path: NDArray[np.float64]  # (n_paths,)
    annualized_return: NDArray[np.float64]  # (n_paths,)
    fair_price: float  # mean PV across paths
    risk: dict[str, float]
    fair_price_ci: tuple[float, float] | None = None  # (lo, hi) bootstrap CI

    def as_record(self) -> dict[str, Any]:
        record: dict[str, Any] = {
            "instrument": self.name,
            "initial_price": float(self.initial_price),
            "fair_price": float(self.fair_price),
            "fair_to_par": float(self.fair_price / self.initial_price)
            if self.initial_price > 0
            else float("nan"),
            "fair_price_lo": float(self.fair_price_ci[0]) if self.fair_price_ci else float("nan"),
            "fair_price_hi": float(self.fair_price_ci[1]) if self.fair_price_ci else float("nan"),
            "fair_price_ci_width": (
                float(self.fair_price_ci[1] - self.fair_price_ci[0])
                if self.fair_price_ci
                else float("nan")
            ),
            "mean_ann_return": float(self.annualized_return.mean()),
            "median_ann_return": float(np.median(self.annualized_return)),
            "prob_negative_return": float((self.annualized_return < 0).mean()),
        }
        record.update({f"risk_{k}": float(v) for k, v in self.risk.items()})
        return record


def _pv_per_path(
    cf: InstrumentCashFlows,
    *,
    discount_factors: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Discounted total cash flow for each path."""
    interest_pv = (cf.interest_cash_flows * discount_factors[:, 1:]).sum(axis=1)
    principal_pv = cf.principal_cash_flow * discount_factors[:, -1]
    return np.asarray(interest_pv + principal_pv, dtype=np.float64)


def bootstrap_fair_price_ci(
    pv_per_path: NDArray[np.float64],
    *,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 20260519,
    chunk: int = 200,
) -> tuple[float, float]:
    """Return ``(lo, hi)`` 1 − ``alpha`` percentile-bootstrap CI for ``E[PV]``.

    Resamples ``n_resamples`` × ``n_paths`` indices with replacement (in
    chunks of ``chunk`` to keep memory bounded) and returns the empirical
    quantiles of the resampled mean.
    """
    arr = np.asarray(pv_per_path, dtype=float)
    n = arr.size
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    written = 0
    while written < n_resamples:
        block = min(chunk, n_resamples - written)
        idx = rng.integers(low=0, high=n, size=(block, n))
        means[written : written + block] = arr[idx].mean(axis=1)
        written += block
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi


def price_instrument(
    cf: InstrumentCashFlows,
    *,
    out: SimulationOutput,
    risk_free_rate: float | None = None,
    omega_threshold: float = 0.0,
    bootstrap_resamples: int = 0,
    bootstrap_seed: int = 20260519,
) -> InstrumentPricing:
    """Run the pricing + risk pipeline for one instrument.

    Parameters
    ----------
    cf
        Per-path cash flows produced by :func:`instruments.extract_all`.
    out
        Underlying ``SimulationOutput`` (needed for discount factors and
        horizon).
    risk_free_rate
        Target return for Sortino's downside deviation. Defaults to the
        empirical mean instantaneous rate observed in the simulation.
    omega_threshold
        Threshold for the Omega ratio (defaults to 0).
    """
    cfg = out.config
    horizon = cfg.horizon_years
    pv = _pv_per_path(cf, discount_factors=out.discount_factors)
    total_return = pv / cf.initial_price - 1.0
    # Annualised geometric realised return; floor at 1e-12 so the power
    # remains finite on paths where the realised PV is non-positive. The
    # diagnostic ``pv_floored_paths`` records how often this floor binds.
    pv_per_par = pv / cf.initial_price
    n_floored = int((pv_per_par <= 1e-12).sum())
    pv_floor = np.maximum(pv_per_par, 1e-12)
    annualised = pv_floor ** (1.0 / horizon) - 1.0

    if risk_free_rate is None:
        risk_free_rate = float(out.rate_paths.mean())

    # Cumulative cash-flow path for drawdown: the mean trajectory across MCs.
    cum_path = np.concatenate(
        [
            np.zeros((1,), dtype=float),
            cf.interest_cash_flows.mean(axis=0).cumsum(),
        ]
    )
    cum_path[-1] += float(cf.principal_cash_flow.mean())

    risk = risk_metrics.summary(
        returns=annualised,
        cumulative_cashflows=cum_path,
        rf=risk_free_rate,
        omega_threshold=omega_threshold,
    )
    risk["mean_total_return"] = float(total_return.mean())
    risk["prob_capital_loss"] = float((total_return < 0).mean())
    risk["pv_floored_paths"] = float(n_floored)

    ci: tuple[float, float] | None = None
    if bootstrap_resamples > 0:
        ci = bootstrap_fair_price_ci(pv, n_resamples=bootstrap_resamples, seed=bootstrap_seed)

    return InstrumentPricing(
        name=cf.name,
        initial_price=float(cf.initial_price),
        total_pv_per_path=pv,
        annualized_return=annualised,
        fair_price=float(pv.mean()),
        risk=risk,
        fair_price_ci=ci,
    )


def price_all(
    instruments: dict[str, InstrumentCashFlows],
    *,
    out: SimulationOutput,
    risk_free_rate: float | None = None,
    bootstrap_resamples: int = 0,
    bootstrap_seed: int = 20260519,
) -> dict[str, InstrumentPricing]:
    """Price every instrument in a comparison batch."""
    return {
        name: price_instrument(
            cf,
            out=out,
            risk_free_rate=risk_free_rate,
            bootstrap_resamples=bootstrap_resamples,
            bootstrap_seed=bootstrap_seed + i,
        )
        for i, (name, cf) in enumerate(instruments.items())
    }


def solve_fair_coupon(
    *,
    out: SimulationOutput,
    tranche_name: str,
    coupons_template: dict[str, float],
    bounds: tuple[float, float] = (0.0, 0.30),
    xtol: float = 1e-5,
) -> float:
    """Solve for the coupon that sets ``PV == par`` for one tranche.

    The default / price / rate paths are held fixed; only the waterfall is
    re-run at each candidate coupon. Brent's method on the residual
    ``PV(c) - par``. When the residual does not change sign across the
    bracket — which can legitimately happen when the rental income is too
    thin to bring the tranche to par for any positive coupon — we return
    ``float("nan")`` so the caller can render that case explicitly rather
    than masking it with a bogus number.
    """
    from ..waterfall import andersen_sidenius

    cfg = out.config
    tranches = _reconstruct_tranches(out)
    par_amount = float(out.waterfall.notional_path[tranche_name][0, 0])

    def residual(c: float) -> float:
        candidate = dict(coupons_template)
        candidate[tranche_name] = c
        wf = andersen_sidenius.run(
            cumulative_loss=out.cumulative_loss,
            net_rent=out.net_rent,
            terminal_value=out.terminal_value,
            tranches=tranches,
            coupons=candidate,
            par=cfg.par,
            dt=out.dt,
        )
        interest_pv = (
            (wf.interest_cash_flows[tranche_name] * out.discount_factors[:, 1:]).sum(axis=1).mean()
        )
        principal_pv = (wf.principal_cash_flows[tranche_name] * out.discount_factors[:, -1]).mean()
        return float(interest_pv + principal_pv - par_amount)

    f_lo = residual(bounds[0])
    f_hi = residual(bounds[1])
    if not (np.isfinite(f_lo) and np.isfinite(f_hi)):
        return float("nan")
    if f_lo * f_hi > 0:
        return float("nan")
    return float(brentq(residual, bounds[0], bounds[1], xtol=xtol))


def solve_fair_coupons_joint(
    *,
    out: SimulationOutput,
    base_coupons: dict[str, float],
    bounds: tuple[tuple[float, float], tuple[float, float]] = ((0.0, 0.30), (0.0, 0.30)),
    initial_guess: tuple[float, float] = (0.04, 0.07),
) -> tuple[float, float]:
    """Solve $(c_S, c_M)$ jointly so both senior and mezzanine PVs equal par.

    Uses ``scipy.optimize.fsolve`` on the 2-D residual vector. Returns
    ``(NaN, NaN)`` whenever the solver fails to converge or the solution
    falls outside the requested bracket. The mezzanine fair coupon will
    generally not exist on the calibrated Paris baseline because the
    available rental cash flow is structurally insufficient — this is
    exactly the empirical finding the working paper documents.
    """
    from ..waterfall import andersen_sidenius

    cfg = out.config
    tranches = _reconstruct_tranches(out)
    senior_par = float(out.waterfall.notional_path["senior"][0, 0])
    mezz_par = float(out.waterfall.notional_path["mezzanine"][0, 0])

    def residual(coupons_vec: NDArray[np.float64]) -> NDArray[np.float64]:
        c_s, c_m = coupons_vec
        candidate = dict(base_coupons)
        candidate["senior"] = float(c_s)
        candidate["mezzanine"] = float(c_m)
        wf = andersen_sidenius.run(
            cumulative_loss=out.cumulative_loss,
            net_rent=out.net_rent,
            terminal_value=out.terminal_value,
            tranches=tranches,
            coupons=candidate,
            par=cfg.par,
            dt=out.dt,
        )
        senior_pv = (wf.interest_cash_flows["senior"] * out.discount_factors[:, 1:]).sum(
            axis=1
        ).mean() + (wf.principal_cash_flows["senior"] * out.discount_factors[:, -1]).mean()
        mezz_pv = (wf.interest_cash_flows["mezzanine"] * out.discount_factors[:, 1:]).sum(
            axis=1
        ).mean() + (wf.principal_cash_flows["mezzanine"] * out.discount_factors[:, -1]).mean()
        return np.array([float(senior_pv - senior_par), float(mezz_pv - mezz_par)], dtype=float)

    try:
        sol, info, ier, _msg = fsolve(
            residual, x0=np.asarray(initial_guess, dtype=float), full_output=True
        )
    except (ValueError, RuntimeError):
        return float("nan"), float("nan")

    if ier != 1:
        return float("nan"), float("nan")
    c_s, c_m = float(sol[0]), float(sol[1])
    if not (bounds[0][0] <= c_s <= bounds[0][1]):
        return float("nan"), float("nan")
    if not (bounds[1][0] <= c_m <= bounds[1][1]):
        return float("nan"), float("nan")
    # Cross-check residual magnitude.
    if np.linalg.norm(info["fvec"]) > 1e-3 * cfg.par:
        return float("nan"), float("nan")
    return c_s, c_m


def solve_fair_coupons_for_all(
    *,
    out: SimulationOutput,
    base_coupons: dict[str, float],
    tranche_names: Sequence[str] = ("senior", "mezzanine"),
    bounds: tuple[float, float] = (0.0, 0.30),
) -> dict[str, float]:
    """Solve fair coupons for every tranche in ``tranche_names`` sequentially.

    Solves the most senior tranche first (then mezzanine), each time using
    the previously-solved coupons in the template so the conditional fair
    value is internally consistent. The equity coupon is left untouched
    (residual claimant).
    """
    fair = dict(base_coupons)
    for name in tranche_names:
        if name not in base_coupons:
            continue
        fair[name] = solve_fair_coupon(
            out=out,
            tranche_name=name,
            coupons_template=fair,
            bounds=bounds,
        )
    return fair


def _reconstruct_tranches(out: SimulationOutput):
    """Read tranche definitions out of a SimulationOutput's waterfall paths.

    The initial notional fraction at t=0 gives the thickness; the attach point
    is the running cumulative thickness across the senior→junior stack.
    """
    from ..waterfall.tranches import Tranche

    cfg = out.config
    items: list[Tranche] = []
    # ``notional_path`` is keyed in junior-to-senior order by construction
    # (the waterfall pre-sorts tranches by attach point before populating
    # the dict). Walk the dict to rebuild the stack for any tranche names.
    attach = 0.0
    for name, notionals in out.waterfall.notional_path.items():
        thickness = float(notionals[0, 0]) / cfg.par
        items.append(Tranche(name=name, attach=attach, detach=attach + thickness))
        attach += thickness
    return items


__all__ = [
    "InstrumentPricing",
    "bootstrap_fair_price_ci",
    "price_all",
    "price_instrument",
    "solve_fair_coupon",
    "solve_fair_coupons_for_all",
    "solve_fair_coupons_joint",
]
