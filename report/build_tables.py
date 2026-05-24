"""Generate LaTeX tables and value macros from the simulation artifacts.

Reads ``artifacts/results.csv``, ``artifacts/results_meta.json`` and
``data/processed/calibrated_params.yaml`` and writes the ``\\input``-friendly
files used by ``report/main.tex``. Run this before compiling the report::

    PYTHONPATH=src python report/build_tables.py

The output files are:

* ``report/tables/calibrated_params.tex`` — calibration table.
* ``report/tables/headline_results.tex`` — per-instrument fair price, return
  and risk metrics.
* ``report/tables/values.tex`` — single-number ``\\newcommand`` macros used
  inline in the section files.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
TABLES = ROOT / "report" / "tables"
CALIBRATED = ROOT / "data" / "processed" / "calibrated_params.yaml"


def _f(v: float, fmt: str) -> str:
    return fmt.format(v)


def make_calibrated_params_table(cal: dict) -> str:
    gbm = cal["gbm_paris"]
    mer = cal["merton_paris"]
    vas = cal["vasicek_oat_10y"]
    return rf"""
\begin{{table}}[ht]
\centering
\caption{{Calibrated parameter point estimates with maximum-likelihood standard errors.}}
\label{{tab:calibration}}
\small
\begin{{tabular}}{{lllllll}}
\toprule
Model & Series & Parameter & Estimate & Std.\ error & LL & AIC \\
\midrule
GBM (Paris)        & Notaires/INSEE & $\mu$        & {_f(gbm['params']['mu'], '{:.4f}')} & {_f(gbm['std_errors']['mu'], '{:.4f}')}    & \multirow{{2}}{{*}}{{{_f(gbm['log_likelihood'], '{:.1f}')}}} & \multirow{{2}}{{*}}{{{_f(gbm['aic'], '{:.1f}')}}} \\
                   &                & $\sigma$     & {_f(gbm['params']['sigma'], '{:.4f}')} & {_f(gbm['std_errors']['sigma'], '{:.4f}')} & & \\
\midrule
Merton             & Notaires/INSEE & $\mu$        & {_f(mer['params']['mu'], '{:.4f}')} &     ---     & \multirow{{5}}{{*}}{{{_f(mer['log_likelihood'], '{:.1f}')}}} & \multirow{{5}}{{*}}{{{_f(mer['aic'], '{:.1f}')}}} \\
                   &                & $\sigma$     & {_f(mer['params']['sigma'], '{:.4f}')} &     ---     & & \\
                   &                & $\lambda$    & {_f(mer['params']['lam'], '{:.4f}')} &     ---     & & \\
                   &                & $\mu_J$      & {_f(mer['params']['mu_jump'], '{:.4f}')} &     ---     & & \\
                   &                & $\sigma_J$   & {_f(mer['params']['sigma_jump'], '{:.4f}')} &     ---     & & \\
\midrule
Vasicek (OAT 10Y)  & FRED IRLTLT01FRM156N & $a$    & {_f(vas['params']['a'], '{:.4f}')} & {_f(vas['std_errors']['a'], '{:.4f}')}    & \multirow{{3}}{{*}}{{{_f(vas['log_likelihood'], '{:.1f}')}}} & \multirow{{3}}{{*}}{{{_f(vas['aic'], '{:.1f}')}}} \\
                   &                      & $b$    & {_f(vas['params']['b'], '{:.4f}')} & {_f(vas['std_errors']['b'], '{:.4f}')} & & \\
                   &                      & $\sigma_r$ & {_f(vas['params']['sigma_r'], '{:.4f}')} & {_f(vas['std_errors']['sigma_r'], '{:.4f}')} & & \\
\bottomrule
\end{{tabular}}
\par\vspace{{0.4em}}\footnotesize\textit{{Sample periods:
Notaires Paris {cal['data_windows']['notaires_paris']['start']}--{cal['data_windows']['notaires_paris']['end']} ($n={cal['data_windows']['notaires_paris']['n']}$);
OAT 10Y {cal['data_windows']['oat_10y']['start']}--{cal['data_windows']['oat_10y']['end']} ($n={cal['data_windows']['oat_10y']['n']}$).}}
\end{{table}}
"""


def make_headline_results_table(df: pd.DataFrame) -> str:
    gauss = df[(df["credit_model"] == "gaussian_copula") & (df["insurance"] == "none")].set_index("instrument")
    rows = []
    label_map = {"model_a": "Model A", "model_b": "Model B", "equity": "Equity", "mezzanine": "Mezzanine", "senior": "Senior"}
    for inst in ["model_a", "model_b", "equity", "mezzanine", "senior"]:
        if inst not in gauss.index:
            continue
        r = gauss.loc[inst]
        rows.append(
            f"{label_map[inst]} & "
            f"{_f(r['initial_price'] / 1e6, '{:.2f}')} & "
            f"{_f(r['fair_price'] / 1e6, '{:.2f}')} & "
            f"{_f(r['fair_to_par'], '{:.3f}')} & "
            f"{_f(r['mean_ann_return'] * 100, '{:.2f}')}\\% & "
            f"{_f(r['risk_std'] * 100, '{:.2f}')}\\% & "
            f"{_f(r['risk_sharpe'], '{:.2f}')} & "
            f"{_f(r['risk_var_95'] * 100, '{:.2f}')}\\% & "
            f"{_f(r['prob_negative_return'] * 100, '{:.1f}')}\\% \\\\"
        )
    body = "\n".join(rows)
    return rf"""
\begin{{table}}[ht]
\centering
\caption{{Headline pricing and risk metrics under the Gaussian one-factor copula baseline, without insurance. All monetary values in millions of euros; returns and volatilities annualised.}}
\label{{tab:headline}}
\small
\begin{{tabular}}{{lrrrrrrrr}}
\toprule
Instrument & Initial price (\EUR M) & Fair price (\EUR M) & Fair / Par & Mean ann.\ ret.\ & Vol. & Sharpe & VaR$_{{95}}$ & Pr.\ negative \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def make_value_macros(df: pd.DataFrame, meta: dict, cal: dict) -> str:
    """Single-number `\\newcommand` macros that section files reference inline."""
    gauss = df[(df["credit_model"] == "gaussian_copula") & (df["insurance"] == "none")].set_index("instrument")
    gbm = cal["gbm_paris"]["params"]
    vas = cal["vasicek_oat_10y"]["params"]

    lines = [
        rf"\newcommand{{\dataParisStart}}{{{cal['data_windows']['notaires_paris']['start']}}}",
        rf"\newcommand{{\dataParisEnd}}{{{cal['data_windows']['notaires_paris']['end']}}}",
        rf"\newcommand{{\dataParisN}}{{{cal['data_windows']['notaires_paris']['n']}}}",
        rf"\newcommand{{\dataOatN}}{{{cal['data_windows']['oat_10y']['n']}}}",
        rf"\newcommand{{\paramGbmMu}}{{{_f(gbm['mu'] * 100, '{:.2f}')}\%}}",
        rf"\newcommand{{\paramGbmSigma}}{{{_f(gbm['sigma'] * 100, '{:.2f}')}\%}}",
        rf"\newcommand{{\paramVasicekA}}{{{_f(vas['a'], '{:.4f}')}}}",
        rf"\newcommand{{\paramVasicekB}}{{{_f(vas['b'] * 100, '{:.2f}')}\%}}",
        rf"\newcommand{{\paramVasicekSigma}}{{{_f(vas['sigma_r'] * 100, '{:.2f}')}\%}}",
        rf"\newcommand{{\parTotal}}{{{_f(meta['par_eur'] / 1e6, '{:.1f}')}~M\EUR}}",
        rf"\newcommand{{\nObligors}}{{{meta.get('n_obligors', 70)}}}",
        rf"\newcommand{{\nPaths}}{{{meta['n_paths']}}}",
        rf"\newcommand{{\horizonYears}}{{{int(meta['horizon_years'])}}}",
        rf"\newcommand{{\pdTerminal}}{{{_f(meta['pd_terminal'] * 100, '{:.1f}')}\%}}",
        rf"\newcommand{{\rhoBaseline}}{{{_f(meta['rho'], '{:.2f}')}}}",
    ]

    for inst in ["model_a", "model_b", "equity", "mezzanine", "senior"]:
        if inst in gauss.index:
            r = gauss.loc[inst]
            cap = inst.replace("_", "")
            macro_name = cap[0].upper() + cap[1:]
            lines.extend([
                rf"\newcommand{{\fairToPar{macro_name}}}{{{_f(r['fair_to_par'], '{:.3f}')}}}",
                rf"\newcommand{{\sharpe{macro_name}}}{{{_f(r['risk_sharpe'], '{:.2f}')}}}",
                rf"\newcommand{{\meanReturn{macro_name}}}{{{_f(r['mean_ann_return'] * 100, '{:.2f}')}\%}}",
                rf"\newcommand{{\probNeg{macro_name}}}{{{_f(r['prob_negative_return'] * 100, '{:.1f}')}\%}}",
            ])

    # Cross-model headline: senior fair_to_par under each credit model.
    senior_per_model = df[(df["instrument"] == "senior") & (df["insurance"] == "none")].set_index("credit_model")
    for model, label in [("gaussian_copula", "Gauss"), ("student_t_copula", "Student"), ("cox_intensity", "Cox")]:
        if model in senior_per_model.index:
            lines.append(
                rf"\newcommand{{\seniorFair{label}}}{{{_f(senior_per_model.loc[model, 'fair_to_par'], '{:.3f}')}}}"
            )

    # Insurance figures.
    if "insurance" in meta and "actuarial" in meta["insurance"]:
        lines.extend([
            rf"\newcommand{{\insuranceAnnualPremium}}{{{_f(meta['insurance']['actuarial']['annual_premium'] / 1000, '{:.0f}')}~k\EUR}}",
            rf"\newcommand{{\insuranceExpectedLoss}}{{{_f(meta['insurance']['actuarial']['expected_loss_per_path_mean'] / 1000, '{:.0f}')}~k\EUR}}",
            rf"\newcommand{{\insuranceCoverage}}{{{_f(meta['insurance']['actuarial']['coverage_cap'] * 100, '{:.0f}')}\%}}",
        ])

    return "\n".join(lines) + "\n"


def make_fair_coupons_table(fair: pd.DataFrame) -> str:
    """LaTeX table of fair coupons per credit model and tranche."""
    # When the CSV carries multiple solvers (sequential + joint), use the
    # sequential rows for the headline table.
    if "solver" in fair.columns:
        fair = fair[fair["solver"] == "sequential"]
    pivot = fair.pivot(index="tranche", columns="credit_model", values="fair_coupon")
    pivot = pivot.reindex(["senior", "mezzanine", "equity"])
    base = fair.set_index(["tranche", "credit_model"])["base_coupon"].unstack()
    models = list(pivot.columns)
    rows = []
    for tr in pivot.index:
        if pd.isna(pivot.loc[tr]).all():
            continue
        label = tr.title()
        contract = "—" if pd.isna(base.loc[tr].iloc[0]) else f"{base.loc[tr].iloc[0] * 100:.2f}\\%"
        cells = [
            f"{val * 100:.2f}\\%" if pd.notna(val) else "—" for val in pivot.loc[tr]
        ]
        rows.append(f"{label} & {contract} & " + " & ".join(cells) + " \\\\")
    body = "\n".join(rows)
    header_models = " & ".join(m.replace("_", " ").title() for m in models)
    return rf"""
\begin{{table}}[ht]
\centering
\caption{{Fair coupons that set tranche PV equal to par at $t = 0$, solved by Brent's method for each credit model. Equity stays as the residual claimant.}}
\label{{tab:fair-coupons}}
\small
\begin{{tabular}}{{l r {' '.join(['r'] * len(models))}}}
\toprule
Tranche & Contract coupon & {header_models} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def make_stress_table(stress: pd.DataFrame, *, credit_model: str = "gaussian_copula") -> str:
    """LaTeX table of fair_to_par per (scenario, instrument) under one credit model."""
    sub = stress[stress["credit_model"] == credit_model]
    pivot = (
        sub.pivot(index="instrument", columns="scenario", values="fair_to_par")
        .reindex(index=["model_a", "model_b", "equity", "mezzanine", "senior"])
    )
    scenarios = list(pivot.columns)
    rows = []
    for inst in pivot.index:
        label = inst.replace("_", " ").title()
        cells = [f"{val:.3f}" if pd.notna(val) else "—" for val in pivot.loc[inst]]
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    body = "\n".join(rows)
    header_scenarios = " & ".join(s.replace("_", " ") for s in scenarios)
    return rf"""
\begin{{table}}[ht]
\centering
\caption{{Per-instrument fair-price ratio across baseline and stress scenarios (Gaussian one-factor copula). Each cell is read from \texttt{{artifacts/stress\_results.csv}}.}}
\label{{tab:stress}}
\small
\begin{{tabular}}{{l {' '.join(['r'] * len(scenarios))}}}
\toprule
Instrument & {header_scenarios} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def make_insurance_comparison_table(df: pd.DataFrame, *, credit_model: str = "gaussian_copula") -> str:
    """LaTeX comparison table — fair_to_par across the three insurance regimes."""
    sub = df[df["credit_model"] == credit_model]
    pivot = sub.pivot(index="instrument", columns="insurance", values="fair_to_par").reindex(
        index=["model_a", "model_b", "equity", "mezzanine", "senior"]
    )
    cols = ["none", "actuarial", "option_theoretic"]
    pivot = pivot[[c for c in cols if c in pivot.columns]]
    rows = []
    for inst in pivot.index:
        label = inst.replace("_", " ").title()
        cells = [f"{val:.3f}" if pd.notna(val) else "—" for val in pivot.loc[inst]]
        rows.append(f"{label} & " + " & ".join(cells) + " \\\\")
    body = "\n".join(rows)
    header = " & ".join(["No insurance", "Actuarial", "Option-theor."][: len(pivot.columns)])
    return rf"""
\begin{{table}}[ht]
\centering
\caption{{Insurance regime comparison: fair price / par per instrument with no insurance, actuarial pricing and option-theoretic pricing (Gaussian one-factor copula baseline).}}
\label{{tab:insurance-comparison}}
\small
\begin{{tabular}}{{l {' '.join(['r'] * len(pivot.columns))}}}
\toprule
Instrument & {header} \\
\midrule
{body}
\bottomrule
\end{{tabular}}
\end{{table}}
"""


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ARTIFACTS / "results.csv")
    meta = json.loads((ARTIFACTS / "results_meta.json").read_text())
    cal = yaml.safe_load(CALIBRATED.read_text())

    # Add n_obligors to meta for the macro generator.
    with (ROOT / "config/paris_intermediate.yaml").open() as fh:
        scenario_yaml = yaml.safe_load(fh)
    parent = scenario_yaml.get("extends")
    if parent is not None:
        with (ROOT / "config" / parent).open() as fh:
            base = yaml.safe_load(fh)
        meta["n_obligors"] = scenario_yaml.get("building", {}).get(
            "n_apartments", base["building"]["n_apartments"]
        )
    else:
        meta["n_obligors"] = scenario_yaml["building"]["n_apartments"]

    (TABLES / "calibrated_params.tex").write_text(make_calibrated_params_table(cal))
    (TABLES / "headline_results.tex").write_text(make_headline_results_table(df))
    (TABLES / "values.tex").write_text(make_value_macros(df, meta, cal))

    fair_path = ARTIFACTS / "fair_coupons.csv"
    if fair_path.exists():
        fair = pd.read_csv(fair_path)
        if not fair.empty:
            (TABLES / "fair_coupons.tex").write_text(make_fair_coupons_table(fair))

    stress_path = ARTIFACTS / "stress_results.csv"
    if stress_path.exists():
        stress = pd.read_csv(stress_path)
        if not stress.empty:
            (TABLES / "stress_results.tex").write_text(make_stress_table(stress))

    (TABLES / "insurance_comparison.tex").write_text(make_insurance_comparison_table(df))

    print(f"Wrote tables to {TABLES}")


if __name__ == "__main__":
    main()
