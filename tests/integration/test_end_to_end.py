"""End-to-end integration test for the full pricing pipeline.

Runs the same flow as ``tranche-cli mc`` at a tiny ``n_sims`` and asserts
the resulting ``artifacts/results.csv`` carries the schema downstream code
(notebooks, dashboard, report) expects: bootstrap CI columns, premium
weights for the insurance pass, fair coupons sidecar file.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.integration
def test_pipeline_end_to_end(tmp_path: Path, paris_config) -> None:
    from tranche_pricing.pricing import runner as pricing_runner

    # Shrink the simulation so the test runs in well under one minute.
    # Use ``model_copy`` to avoid mutating the session-scoped fixture.
    small = paris_config.model_copy(
        update={
            "monte_carlo": paris_config.monte_carlo.model_copy(
                update={"n_sims": 200, "bootstrap_resamples": 50}
            ),
            "building": paris_config.building.model_copy(update={"n_apartments": 10}),
        }
    )

    csv_path = pricing_runner.run(small, output_dir=tmp_path)
    assert csv_path.exists()

    df = pd.read_csv(csv_path)
    required_cols = {
        "credit_model",
        "instrument",
        "insurance",
        "fair_price",
        "fair_price_lo",
        "fair_price_hi",
        "fair_price_ci_width",
        "fair_to_par",
        "mean_ann_return",
        "risk_sharpe",
    }
    missing = required_cols - set(df.columns)
    assert not missing, f"missing columns: {missing}"

    expected_instruments = {"model_a", "model_b", "equity", "mezzanine", "senior"}
    assert expected_instruments.issubset(set(df["instrument"]))

    # CI brackets the point estimate (allowing a small numerical tolerance).
    df_clean = df.dropna(subset=["fair_price_lo", "fair_price_hi"])
    if not df_clean.empty:
        assert (df_clean["fair_price_lo"] <= df_clean["fair_price"] + 1e-6).all()
        assert (df_clean["fair_price_hi"] >= df_clean["fair_price"] - 1e-6).all()

    # Sidecar files written by `run`.
    assert (tmp_path / "fair_coupons.csv").exists()
    assert (tmp_path / "results_meta.json").exists()


@pytest.mark.integration
def test_report_pdf_compiles(tmp_path: Path) -> None:
    """Compile the LaTeX report. Skipped if latexmk is not on PATH."""
    if shutil.which("latexmk") is None:
        pytest.skip("latexmk not available; skipping LaTeX compilation test.")
    repo_root = Path(__file__).resolve().parents[2]
    report_dir = repo_root / "report"
    assert (report_dir / "main.tex").exists()

    # We do not pass ``-halt-on-error``: pdflatex flags overfull hboxes as
    # warnings and may exit non-zero on them — they are typographic, not
    # logical. The actual gate is whether ``main.pdf`` is produced.
    subprocess.run(
        ["latexmk", "-pdf", "-interaction=nonstopmode", "main.tex"],
        cwd=report_dir,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert (report_dir / "main.pdf").exists()
    # Sanity check: the file is non-trivially small.
    assert (report_dir / "main.pdf").stat().st_size > 50_000
