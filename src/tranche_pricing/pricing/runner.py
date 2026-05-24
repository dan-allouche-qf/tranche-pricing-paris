"""End-to-end pricing pipeline driven by the CLI ``mc`` subcommand.

Loads ``data/processed/calibrated_params.yaml`` produced by
:mod:`tranche_pricing.calibration.runner`, builds the parameter dataclasses,
runs the joint Gaussian/Student-t/Cox simulation, prices the insurance,
serialises everything into ``artifacts/results.csv`` plus a JSON sidecar
``artifacts/results_meta.json``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ..config import Config
from ..credit import cox_intensity
from ..credit.lgd import BetaLGDParams
from ..data import oat
from ..data.cache import PROCESSED_DIR
from ..insurance import actuarial, option_theoretic
from ..markets.price_gbm import GBMParams
from ..markets.rates_vasicek import VasicekParams
from ..simulation import SimulationConfig
from ..waterfall.tranches import Tranche
from .model_compare import compare_credit_models

logger = logging.getLogger(__name__)


def _recent_oat_level(fallback: float = 0.030) -> float:
    """Most recent observed OAT 10Y level (in decimal, e.g. 0.03 = 3%).

    Uses the snapshot in ``data/raw/oat_10y.csv`` if it exists. Returns
    ``fallback`` when the snapshot is missing (e.g. on first install).
    """
    try:
        df = oat.fetch()
    except FileNotFoundError:
        return float(fallback)
    if df.empty:
        return float(fallback)
    return float(df["yield_pct"].dropna().iloc[-1]) / 100.0


def build_inputs_from_yaml(
    cfg: Config,
    calibrated_path: Path | None = None,
) -> tuple[
    SimulationConfig, GBMParams, VasicekParams, BetaLGDParams, list[Tranche], dict[str, float]
]:
    """Build the simulation inputs from a project YAML + calibrated_params.yaml."""
    path = (
        calibrated_path if calibrated_path is not None else PROCESSED_DIR / "calibrated_params.yaml"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Calibrated parameters not found at {path}. Run `make calibrate` first."
        )
    with path.open() as fh:
        cal = yaml.safe_load(fh)

    gbm = GBMParams(
        mu=float(cal["gbm_paris"]["params"]["mu"]),
        sigma=float(cal["gbm_paris"]["params"]["sigma"]),
    )
    vas = VasicekParams(
        a=float(cal["vasicek_oat_10y"]["params"]["a"]),
        b=float(cal["vasicek_oat_10y"]["params"]["b"]),
        sigma_r=float(cal["vasicek_oat_10y"]["params"]["sigma_r"]),
    )
    lgd = BetaLGDParams(
        mean=float(cfg.lgd.mean_init),
        std=float(cfg.lgd.std_init),
    )

    n_steps = cfg.building.steps_per_year
    par = (
        cfg.building.n_apartments
        * cfg.building.surface_per_apt_m2
        * cfg.building.initial_price_per_m2_eur
    )
    # Initial short rate for discounting. When the user picks
    # ``constant_at_initial_rate`` we start at the most recent OAT 10Y level
    # observed in the data (close to 3% in 2025) so the discount factors
    # reflect current conditions rather than the secular long-run mean.
    use_constant = cfg.risk.discount_convention == "constant_at_initial_rate"
    initial_rate = _recent_oat_level() if use_constant else vas.b

    # If the CIR macro factor has been calibrated on the unemployment
    # series, build a CoxIntensityParams whose (kappa, theta, xi) come from
    # the data; alpha (idiosyncratic) and beta (factor loading) stay at
    # their YAML priors. Otherwise leave cox=None so the engine falls back
    # on the analytical alpha calibration in
    # ``tranche_pricing.credit.cox_intensity.calibrate_alpha_for_pd``.
    cox_params: cox_intensity.CoxIntensityParams | None = None
    if "cir_unemployment" in cal:
        cir = cal["cir_unemployment"]["params"]
        beta = float(cfg.credit.cox_intensity.beta_init)
        theta = float(cir["theta"])
        alpha = cox_intensity.calibrate_alpha_for_pd(
            pd_terminal=min(
                0.95, 1.0 - (1.0 - cfg.credit.pd_annual_init) ** cfg.building.horizon_years
            ),
            horizon_years=cfg.building.horizon_years,
            beta=beta,
            theta=theta,
        )
        cox_params = cox_intensity.CoxIntensityParams(
            alpha=alpha,
            beta=beta,
            kappa=float(cir["kappa"]),
            theta=theta,
            xi=float(cir["xi"]),
        )

    sim = SimulationConfig(
        n_obligors=cfg.building.n_apartments,
        horizon_years=cfg.building.horizon_years,
        steps_per_year=n_steps,
        par=par,
        gross_yield=cfg.rent.gross_yield_initial,
        maintenance_pct=cfg.rent.maintenance_pct_of_value,
        initial_rate=initial_rate,
        pd_terminal=min(
            0.95,
            1.0 - (1.0 - cfg.credit.pd_annual_init) ** cfg.building.horizon_years,
        ),
        credit_model=cfg.credit.default_model,
        rho=cfg.credit.gaussian_copula.rho_init,
        nu=cfg.credit.student_t_copula.nu_grid[0] if cfg.credit.student_t_copula.nu_grid else 5.0,
        cox=cox_params,
        n_paths=cfg.monte_carlo.n_sims,
        master_seed=cfg.monte_carlo.seed,
        antithetic=cfg.monte_carlo.antithetic,
        use_qmc=cfg.monte_carlo.qmc.enabled,
        discount_convention=cfg.risk.discount_convention,
    )

    tranches = sorted(
        [Tranche(t.name, t.attach, t.detach) for t in cfg.tranches],
        key=lambda t: t.attach,
    )
    # Default coupons: senior 3 %, mezzanine 6 %, equity 0 (residual claim).
    coupons = {
        "senior": 0.03,
        "mezzanine": 0.06,
        "equity": 0.0,
    }
    return sim, gbm, vas, lgd, tranches, coupons


def run(cfg: Config, *, output_dir: Path | None = None) -> Path:
    """Run the comparison + insurance pipeline and write ``results.csv``."""
    sim, gbm, vas, lgd, tranches, coupons = build_inputs_from_yaml(cfg)
    logger.info(
        "Simulation: %s scenario, n_obligors=%d, n_sims=%d, par=%.0f EUR",
        cfg.scenario.name,
        sim.n_obligors,
        sim.n_paths,
        sim.par,
    )

    boot_n = int(cfg.monte_carlo.bootstrap_resamples)

    # Pass 1: without insurance.
    df_noins, outputs = compare_credit_models(
        sim,
        gbm=gbm,
        vasicek=vas,
        lgd=lgd,
        tranches=tranches,
        coupons=coupons,
        bootstrap_resamples=boot_n,
    )
    df_noins["insurance"] = "none"

    # Pass 1bis: solve fair coupons under each credit model. We report both
    # the sequential solver (senior first, then mezzanine) and the joint
    # 2-D solver — the report's discussion compares the two.
    from .tranche_pricer import solve_fair_coupons_for_all, solve_fair_coupons_joint

    fair_coupon_rows: list[dict[str, Any]] = []
    for model_name, model_out in outputs.items():
        fair_seq = solve_fair_coupons_for_all(
            out=model_out,
            base_coupons=coupons,
            tranche_names=("senior", "mezzanine"),
        )
        for tranche_name, c in fair_seq.items():
            fair_coupon_rows.append(
                {
                    "credit_model": model_name,
                    "tranche": tranche_name,
                    "solver": "sequential",
                    "fair_coupon": float(c),
                    "base_coupon": float(coupons[tranche_name]),
                }
            )
        c_s, c_m = solve_fair_coupons_joint(out=model_out, base_coupons=coupons)
        for tranche_name, c in (("senior", c_s), ("mezzanine", c_m), ("equity", 0.0)):
            fair_coupon_rows.append(
                {
                    "credit_model": model_name,
                    "tranche": tranche_name,
                    "solver": "joint",
                    "fair_coupon": float(c),
                    "base_coupon": float(coupons[tranche_name]),
                }
            )

    # Pass 2: with insurance — subtract actuarial premium from each period's rent
    # and re-run pricing on the Gaussian baseline output for the comparison.
    baseline = outputs["gaussian_copula"]
    act = actuarial.price_premium(
        baseline.cumulative_loss,
        par=sim.par,
        coverage_cap=cfg.insurance.coverage_cap,
        admin_loading=cfg.insurance.actuarial.admin_loading,
        risk_loading=cfg.insurance.actuarial.risk_loading,
        horizon_years=sim.horizon_years,
    )
    opt = option_theoretic.price_premium(
        baseline,
        deductible_months=cfg.insurance.option_theoretic.deductible_months,
        coverage_cap=cfg.insurance.coverage_cap,
    )

    # Insurance scenario: re-run with premium subtracted from net_rent.
    sim_ins = replace(sim, master_seed=sim.master_seed + 1)  # decorrelate seeds
    annual_premium = act["annual_premium"]
    # The insurance reduces effective loss (we model it as covering coverage_cap of loss)
    df_ins, _outputs_ins = _price_with_insurance(
        sim_ins,
        gbm=gbm,
        vasicek=vas,
        lgd=lgd,
        tranches=tranches,
        coupons=coupons,
        annual_premium=annual_premium,
        coverage_cap=cfg.insurance.coverage_cap,
        bootstrap_resamples=boot_n,
    )
    df_ins["insurance"] = "actuarial"

    # Option-theoretic pricing variant: same coverage, premium derived from
    # the put-on-rent-shortfall valuation. Decorrelate the seed once more.
    sim_opt = replace(sim, master_seed=sim.master_seed + 2)
    df_opt, _opt_outputs = _price_with_insurance(
        sim_opt,
        gbm=gbm,
        vasicek=vas,
        lgd=lgd,
        tranches=tranches,
        coupons=coupons,
        annual_premium=opt["annual_premium"],
        coverage_cap=cfg.insurance.coverage_cap,
        bootstrap_resamples=boot_n,
    )
    df_opt["insurance"] = "option_theoretic"

    # Assemble.
    results = pd.concat([df_noins, df_ins, df_opt], axis=0, ignore_index=True)
    results["scenario"] = cfg.scenario.name
    results["n_paths"] = sim.n_paths

    out_dir = output_dir if output_dir is not None else Path("artifacts")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "results.csv"
    results.to_csv(csv_path, index=False)

    fair_coupons_df = pd.DataFrame(fair_coupon_rows)
    fair_coupons_df.to_csv(out_dir / "fair_coupons.csv", index=False)

    # Side artefact for the insurance comparison section: a tidy table that
    # makes the no-insurance vs actuarial vs option-theoretic rows easy to
    # consume from the report and the dashboard.
    insurance_comparison_cols = [
        "credit_model",
        "instrument",
        "insurance",
        "fair_price",
        "fair_to_par",
        "mean_ann_return",
        "risk_sharpe",
        "risk_var_95",
    ]
    results[insurance_comparison_cols].to_csv(out_dir / "insurance_comparison.csv", index=False)

    # Stress replay artefact for the stress backtest section of the report.
    from .stress import run_stress_replays

    try:
        stress_df = run_stress_replays(
            cfg,
            base_sim=sim,
            gbm=gbm,
            vasicek=vas,
            lgd=lgd,
            tranches=tranches,
            coupons=coupons,
        )
        stress_df.to_csv(out_dir / "stress_results.csv", index=False)
        logger.info("Wrote %s (%d rows)", out_dir / "stress_results.csv", len(stress_df))
    except Exception as exc:  # pragma: no cover - non-fatal
        logger.warning("Stress replay failed: %s", exc)

    fair_coupons_by_model: dict[str, dict[str, float]] = {}
    for row in fair_coupon_rows:
        fair_coupons_by_model.setdefault(row["credit_model"], {})[row["tranche"]] = float(
            row["fair_coupon"]
        )

    meta = {
        "scenario": cfg.scenario.name,
        "n_paths": sim.n_paths,
        "par_eur": sim.par,
        "horizon_years": sim.horizon_years,
        "pd_terminal": sim.pd_terminal,
        "rho": sim.rho,
        "initial_rate": sim.initial_rate,
        "discount_convention": sim.discount_convention,
        "gbm_params": asdict(gbm),
        "vasicek_params": asdict(vas),
        "lgd_params": asdict(lgd),
        "insurance": {"actuarial": act, "option_theoretic": opt},
        "fair_coupons": fair_coupons_by_model,
    }
    (out_dir / "results_meta.json").write_text(json.dumps(meta, indent=2, default=float))

    logger.info("Wrote %s (%d rows)", csv_path, len(results))
    return csv_path


def _price_with_insurance(
    sim: SimulationConfig,
    *,
    gbm: GBMParams,
    vasicek: VasicekParams,
    lgd: BetaLGDParams,
    tranches: list[Tranche],
    coupons: dict[str, float],
    annual_premium: float,
    coverage_cap: float,
    bootstrap_resamples: int = 0,
) -> tuple[pd.DataFrame, dict]:
    """Re-run comparison with insurance.

    Insurance covers ``coverage_cap`` of every realised default loss, so
    ``cumulative_loss`` is reduced by ``(1 - coverage_cap)``. The premium
    is allocated across tranches in proportion to each tranche's expected
    covered loss under the no-insurance baseline, so the tranches that
    benefit most from coverage pay the largest share of the premium. The
    premium is then subtracted from each tranche's interest cash flow
    post-waterfall. Model A and Model B instruments share the residual
    premium drag pro-rata to their expected apartment-level default
    exposure.
    """
    from copy import deepcopy

    from ..insurance.actuarial import allocate_premium_to_tranches
    from ..pricing.instruments import extract_all
    from ..pricing.tranche_pricer import price_all
    from ..simulation import run_simulation
    from ..waterfall import andersen_sidenius

    rows = []
    outputs = {}
    for model in ("gaussian_copula", "student_t_copula", "cox_intensity"):
        cfg_m = replace(sim, credit_model=model)
        out_no = run_simulation(
            cfg_m,
            gbm=gbm,
            vasicek=vasicek,
            lgd=lgd,
            tranches=tranches,
            coupons=coupons,
        )

        # Premium weights derived from the NO-INSURANCE cumulative loss
        # so the allocation reflects who *would* have absorbed the loss.
        weights = allocate_premium_to_tranches(
            cumulative_loss_no_ins=out_no.cumulative_loss,
            tranches=tranches,
            coverage_cap=coverage_cap,
        )

        out2 = deepcopy(out_no)
        out2.cumulative_loss = out_no.cumulative_loss * (1.0 - coverage_cap)
        out2.terminal_value = (1.0 - out2.cumulative_loss[:, -1]) * out_no.price_paths[:, -1]
        dt = float(out_no.dt)
        # No global premium drag on net_rent — each tranche pays its share
        # post-waterfall (see below).
        out2.net_rent = out_no.net_rent.copy()
        out2.waterfall = andersen_sidenius.run(
            cumulative_loss=out2.cumulative_loss,
            net_rent=out2.net_rent,
            terminal_value=out2.terminal_value,
            tranches=tranches,
            coupons=coupons,
            par=sim.par,
            dt=dt,
        )

        # Per-tranche premium drag, applied to the interest cash flows.
        for tr in tranches:
            share = weights.get(tr.name, 0.0)
            premium_per_period = share * annual_premium * dt
            out2.waterfall.interest_cash_flows[tr.name] = (
                out2.waterfall.interest_cash_flows[tr.name] - premium_per_period
            )

        outputs[model] = out2
        instruments = extract_all(out2)

        # Model A and Model B exposure to the premium: each carries the
        # proportional share its single-apartment / pooled cash flow has
        # in the aggregate. We charge them at par/n_obligors of the
        # aggregate annual premium per period.
        n_obl = sim.n_obligors
        share_ma_mb = annual_premium * dt / n_obl
        instruments["model_a"].interest_cash_flows[:] = (
            instruments["model_a"].interest_cash_flows - share_ma_mb
        )
        instruments["model_b"].interest_cash_flows[:] = (
            instruments["model_b"].interest_cash_flows - share_ma_mb
        )

        pricings = price_all(instruments, out=out2, bootstrap_resamples=bootstrap_resamples)
        for inst_name, pricing in pricings.items():
            record = pricing.as_record()
            record["credit_model"] = model
            record["instrument"] = inst_name
            record["premium_weight"] = float(weights.get(inst_name, 0.0))
            rows.append(record)

    df = pd.DataFrame(rows)
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


__all__ = ["build_inputs_from_yaml", "run"]
