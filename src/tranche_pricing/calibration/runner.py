"""End-to-end calibration runner used by the CLI and notebook 02.

Fetches Notaires-INSEE, IRL and OAT 10Y data through the data layer, runs the
MLEs for GBM, Merton jump-diffusion and Vasicek, and serialises the fitted
parameters plus diagnostics to ``data/processed/calibrated_params.yaml`` so
the simulation engine can read them without re-running calibration.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ..data import insee_irl, insee_unemployment, notaires, oat
from ..data.cache import PROCESSED_DIR
from . import cox_calibrate, mle_gbm, mle_jump, mle_vasicek
from ._types import FitResult

logger = logging.getLogger(__name__)


def _python_native(value: Any) -> Any:
    """Recursively cast numpy scalars and dataclasses to YAML-friendly types."""
    if is_dataclass(value) and not isinstance(value, type):
        return _python_native(asdict(value))
    if isinstance(value, dict):
        return {k: _python_native(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_python_native(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _result_to_dict(result: FitResult[Any]) -> dict[str, Any]:
    payload: dict[str, Any] = _python_native(
        {
            "params": result.params,
            "std_errors": result.std_errors,
            "log_likelihood": float(result.log_likelihood),
            "n_obs": int(result.n_obs),
            "aic": float(result.aic),
            "bic": float(result.bic),
            "extra": dict(result.extra),
        }
    )
    return payload


def run_all() -> dict[str, Any]:
    """Run every calibration and return a dict ready to serialise to YAML."""
    paris = notaires.fetch()
    oat_df = oat.fetch()
    try:
        irl_df = insee_irl.fetch()
    except FileNotFoundError:
        irl_df = None
        logger.warning("IRL snapshot missing; skipping rent-indexation diagnostics.")
    try:
        unemp_df = insee_unemployment.fetch()
    except FileNotFoundError:
        unemp_df = None
        logger.warning("Unemployment snapshot missing; skipping Cox CIR calibration.")

    log_returns_paris = np.diff(np.log(paris["price_index"].dropna().values))
    fit_gbm = mle_gbm.calibrate(log_returns_paris, dt=0.25)
    fit_merton = mle_jump.calibrate(log_returns_paris, dt=0.25)
    fit_vasicek = mle_vasicek.calibrate(
        oat_df["yield_pct"].dropna().values / 100.0,
        dt=1.0 / 12.0,
    )

    payload: dict[str, Any] = {
        "gbm_paris": _result_to_dict(fit_gbm),
        "merton_paris": _result_to_dict(fit_merton),
        "vasicek_oat_10y": _result_to_dict(fit_vasicek),
        "data_windows": {
            "notaires_paris": {
                "start": str(paris["date"].iloc[0].date()),
                "end": str(paris["date"].iloc[-1].date()),
                "n": len(paris),
            },
            "oat_10y": {
                "start": str(oat_df["date"].iloc[0].date()),
                "end": str(oat_df["date"].iloc[-1].date()),
                "n": len(oat_df),
            },
        },
    }
    if irl_df is not None:
        payload["data_windows"]["insee_irl"] = {
            "start": str(irl_df["date"].iloc[0].date()),
            "end": str(irl_df["date"].iloc[-1].date()),
            "n": len(irl_df),
        }

    # Cox CIR calibration on the French ILO unemployment rate (quarterly).
    # The unemployment level is in percent; we divide by 100 to obtain a
    # decimal scale comparable to a hazard intensity.
    if unemp_df is not None and len(unemp_df) >= 20:
        unemp_series = unemp_df["unemployment_pct"].dropna().values / 100.0
        fit_cir = cox_calibrate.calibrate(unemp_series, dt=0.25)
        payload["cir_unemployment"] = _result_to_dict(fit_cir)
        payload["data_windows"]["insee_unemployment"] = {
            "start": str(unemp_df["date"].iloc[0].date()),
            "end": str(unemp_df["date"].iloc[-1].date()),
            "n": len(unemp_df),
        }

    return payload


def persist(payload: dict[str, Any], path: Path | None = None) -> Path:
    """Write the calibration payload to ``data/processed/calibrated_params.yaml``."""
    out = path if path is not None else PROCESSED_DIR / "calibrated_params.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        yaml.safe_dump(payload, fh, sort_keys=True, default_flow_style=False)
    logger.info("Wrote calibrated params to %s", out)
    return out


__all__ = ["persist", "run_all"]
