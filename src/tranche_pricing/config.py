"""Schema-validated configuration loader.

Each YAML in ``config/`` describes a scenario. Files may declare ``extends:
<other.yaml>`` to inherit from a parent file (which is loaded recursively); the
child's values override the parent's, with deep merging for nested mappings.

The resulting dictionary is validated against the Pydantic models below so the
rest of the codebase can treat configuration as a typed object instead of a raw
dict.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# --------------------------------------------------------------------------- #
# Building / cash-flow structural blocks                                      #
# --------------------------------------------------------------------------- #


class BuildingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_apartments: int = Field(..., ge=1)
    surface_per_apt_m2: float = Field(..., gt=0)
    horizon_years: int = Field(..., ge=1)
    steps_per_year: int = Field(..., ge=1)
    initial_price_per_m2_eur: float = Field(..., gt=0)


class RentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gross_yield_initial: float = Field(..., gt=0, lt=1)
    rent_indexation: Literal["irl", "none", "cpi"] = "irl"
    vacancy_rate: float = Field(..., ge=0, lt=1)
    maintenance_pct_of_value: float = Field(..., ge=0, lt=1)


# --------------------------------------------------------------------------- #
# Market dynamics                                                             #
# --------------------------------------------------------------------------- #


class MertonInit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    lambda_init: float = Field(..., gt=0)
    mu_jump_init: float
    sigma_jump_init: float = Field(..., gt=0)


class PriceDynamicsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Literal["gbm", "merton"] = "gbm"
    mu_init: float
    sigma_init: float = Field(..., gt=0)
    merton: MertonInit


class RatesConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: Literal["vasicek", "hull_white"] = "vasicek"
    a_init: float = Field(..., gt=0)
    b_init: float
    sigma_r_init: float = Field(..., gt=0)


# --------------------------------------------------------------------------- #
# Credit                                                                      #
# --------------------------------------------------------------------------- #


class GaussianCopulaInit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rho_init: float = Field(..., ge=0, lt=1)
    rho_grid: list[float] = Field(default_factory=list)

    @field_validator("rho_grid")
    @classmethod
    def _check_rho_grid(cls, value: list[float]) -> list[float]:
        for rho in value:
            if not 0 <= rho < 1:
                raise ValueError(f"rho out of range: {rho}")
        return value


class StudentTCopulaInit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    rho_init: float = Field(..., ge=0, lt=1)
    nu_grid: list[float] = Field(default_factory=list)

    @field_validator("nu_grid")
    @classmethod
    def _check_nu(cls, value: list[float]) -> list[float]:
        for nu in value:
            if nu <= 2:
                raise ValueError(f"Student-t nu must be > 2 (variance defined): {nu}")
        return value


class CoxIntensityInit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    alpha_init: float = Field(..., ge=0)
    beta_init: float = Field(..., ge=0)
    kappa_init: float = Field(..., gt=0)
    theta_init: float = Field(..., gt=0)
    xi_init: float = Field(..., gt=0)


class CreditConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    models: list[Literal["gaussian_copula", "student_t_copula", "cox_intensity"]]
    default_model: Literal["gaussian_copula", "student_t_copula", "cox_intensity"]
    pd_annual_init: float = Field(..., gt=0, lt=1)
    gaussian_copula: GaussianCopulaInit
    student_t_copula: StudentTCopulaInit
    cox_intensity: CoxIntensityInit

    @model_validator(mode="after")
    def _default_model_in_models(self) -> CreditConfig:
        if self.default_model not in self.models:
            raise ValueError(
                f"default_model={self.default_model!r} must be in models={self.models!r}"
            )
        return self


# --------------------------------------------------------------------------- #
# LGD                                                                         #
# --------------------------------------------------------------------------- #


class LgdConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    distribution: Literal["beta", "fixed"] = "beta"
    mean_init: float = Field(..., gt=0, lt=1)
    std_init: float = Field(..., gt=0)

    @model_validator(mode="after")
    def _beta_feasible(self) -> LgdConfig:
        # For a Beta on (0,1): var must be < mean * (1 - mean)
        if self.distribution == "beta":
            mean, var = self.mean_init, self.std_init**2
            if var >= mean * (1 - mean):
                raise ValueError(
                    f"Beta with mean={mean} cannot have std={self.std_init}; "
                    f"variance must be < mean*(1-mean) = {mean * (1 - mean):.4f}"
                )
        return self


# --------------------------------------------------------------------------- #
# Tranches & waterfall                                                        #
# --------------------------------------------------------------------------- #


class TrancheConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    attach: float = Field(..., ge=0, le=1)
    detach: float = Field(..., gt=0, le=1)

    @model_validator(mode="after")
    def _ordered(self) -> TrancheConfig:
        if self.attach >= self.detach:
            raise ValueError(
                f"Tranche {self.name!r}: attach={self.attach} must be < detach={self.detach}"
            )
        return self


class WaterfallConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["andersen_sidenius", "annual_reset"] = "andersen_sidenius"
    oc_test_enabled: bool = False


# --------------------------------------------------------------------------- #
# Insurance                                                                   #
# --------------------------------------------------------------------------- #


class ActuarialConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    admin_loading: float = Field(..., ge=0)
    risk_loading: float = Field(..., ge=0)


class OptionTheoreticConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    deductible_months: int = Field(..., ge=0)


class InsuranceConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled_scenarios: list[bool]
    actuarial: ActuarialConfig
    option_theoretic: OptionTheoreticConfig
    coverage_cap: float = Field(..., gt=0, le=1)


# --------------------------------------------------------------------------- #
# Monte Carlo / risk / data window                                            #
# --------------------------------------------------------------------------- #


class QMCConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    engine: Literal["sobol", "halton"] = "sobol"
    scramble: bool = True


class MonteCarloConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_sims: int = Field(..., ge=1)
    seed: int
    antithetic: bool = True
    control_variates: bool = True
    qmc: QMCConfig
    bootstrap_resamples: int = Field(..., ge=0)


class RiskConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    var_alpha: list[float]
    es_alpha: list[float]
    sortino_target: Literal["risk_free", "zero"] = "risk_free"
    omega_threshold: float = 0.0
    benchmark_series: str = "notaires_paris"
    discount_convention: Literal["vasicek_simulated", "constant_at_initial_rate"] = (
        "vasicek_simulated"
    )


class DataWindow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: str
    end: str


class ScenarioMeta(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str = ""


# --------------------------------------------------------------------------- #
# Stress overlays                                                             #
# --------------------------------------------------------------------------- #


class StressPriceOverlay(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mu_shift_pct: float = 0.0
    sigma_multiplier: float = 1.0
    duration_years: int = Field(..., ge=0)


class StressCreditOverlay(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pd_multiplier: float = Field(..., gt=0)
    pd_lag_years: int = 0
    duration_years: int = Field(..., ge=0)


class StressRatesOverlay(BaseModel):
    model_config = ConfigDict(extra="forbid")

    delta_bps: float = 0.0
    duration_years: int = Field(..., ge=0)


class StressRentOverlay(BaseModel):
    model_config = ConfigDict(extra="forbid")

    moratorium_pct: float = Field(..., ge=0, le=1)
    duration_years: int = Field(..., ge=0)


class StressOverlay(BaseModel):
    model_config = ConfigDict(extra="forbid")

    price_dynamics: StressPriceOverlay | None = None
    credit: StressCreditOverlay | None = None
    rates: StressRatesOverlay | None = None
    rent: StressRentOverlay | None = None


class StressReporting(BaseModel):
    model_config = ConfigDict(extra="forbid")

    highlight_tail_in_figures: bool = False


# --------------------------------------------------------------------------- #
# Top-level config                                                            #
# --------------------------------------------------------------------------- #


class Config(BaseModel):
    """Top-level configuration object loaded from YAML."""

    model_config = ConfigDict(extra="forbid")

    scenario: ScenarioMeta
    building: BuildingConfig
    rent: RentConfig
    price_dynamics: PriceDynamicsConfig
    rates: RatesConfig
    credit: CreditConfig
    lgd: LgdConfig
    tranches: list[TrancheConfig]
    waterfall: WaterfallConfig
    insurance: InsuranceConfig
    monte_carlo: MonteCarloConfig
    risk: RiskConfig
    data_window: DataWindow

    # Optional stress overlays / reporting (only present in stress configs).
    overlay: StressOverlay | None = None
    reporting: StressReporting | None = None

    @model_validator(mode="after")
    def _check_tranches(self) -> Config:
        sorted_t = sorted(self.tranches, key=lambda t: t.attach)
        if sorted_t[0].attach != 0.0:
            raise ValueError("Tranches must start at attach=0.")
        if sorted_t[-1].detach != 1.0:
            raise ValueError("Tranches must cover up to detach=1.")
        for prev, nxt in itertools.pairwise(sorted_t):
            if prev.detach != nxt.attach:
                raise ValueError(f"Tranche stack has a gap between {prev.name!r} and {nxt.name!r}")
        return self


# --------------------------------------------------------------------------- #
# Loader (with `extends:` resolution)                                         #
# --------------------------------------------------------------------------- #


def _deep_merge(parent: dict[str, Any], child: dict[str, Any]) -> dict[str, Any]:
    """Return ``parent | child`` with dict values merged recursively."""
    out = dict(parent)
    for key, val in child.items():
        if key in out and isinstance(out[key], dict) and isinstance(val, dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def _load_raw(path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    """Load a YAML config and resolve any ``extends`` chain."""
    seen = seen or set()
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"Cyclic extends detected involving {resolved}")
    seen.add(resolved)

    with resolved.open() as fh:
        raw: dict[str, Any] = yaml.safe_load(fh)

    parent_ref = raw.pop("extends", None)
    if parent_ref is None:
        return raw

    parent_path = (resolved.parent / parent_ref).resolve()
    parent = _load_raw(parent_path, seen)
    return _deep_merge(parent, raw)


def load_config(path: str | Path) -> Config:
    """Parse the given YAML, resolve ``extends`` and validate against ``Config``."""
    raw = _load_raw(Path(path))
    return Config.model_validate(raw)


__all__ = [
    "BuildingConfig",
    "Config",
    "CreditConfig",
    "DataWindow",
    "InsuranceConfig",
    "LgdConfig",
    "MonteCarloConfig",
    "PriceDynamicsConfig",
    "RatesConfig",
    "RentConfig",
    "RiskConfig",
    "ScenarioMeta",
    "StressOverlay",
    "StressReporting",
    "TrancheConfig",
    "WaterfallConfig",
    "load_config",
]
