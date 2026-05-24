"""Sanity tests for the YAML configuration loader."""

from __future__ import annotations

from pathlib import Path

import pytest

from tranche_pricing.config import (
    Config,
    LgdConfig,
    TrancheConfig,
    _deep_merge,
    load_config,
)

# --------------------------------------------------------------------------- #
# All shipped YAMLs validate                                                  #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "name",
    [
        "default.yaml",
        "paris_intermediate.yaml",
        "stress_gfc.yaml",
        "stress_covid.yaml",
        "stress_rates2022.yaml",
    ],
)
def test_shipped_yaml_loads(config_dir: Path, name: str) -> None:
    cfg = load_config(config_dir / name)
    assert isinstance(cfg, Config)
    assert cfg.scenario.name


# --------------------------------------------------------------------------- #
# `extends:` inheritance                                                      #
# --------------------------------------------------------------------------- #


def test_paris_inherits_from_default(paris_config: Config) -> None:
    # Paris overrides the building scale but does not redefine the credit
    # block, so the inherited models must come through unchanged.
    assert paris_config.building.n_apartments == 70
    assert "gaussian_copula" in paris_config.credit.models


def test_stress_overlays_present(config_dir: Path) -> None:
    gfc = load_config(config_dir / "stress_gfc.yaml")
    assert gfc.overlay is not None
    assert gfc.overlay.price_dynamics is not None
    assert gfc.overlay.price_dynamics.mu_shift_pct == pytest.approx(-0.08)
    assert gfc.overlay.credit is not None
    assert gfc.overlay.credit.pd_multiplier == pytest.approx(2.0)


def test_deep_merge_overrides_nested() -> None:
    parent = {"a": {"x": 1, "y": 2}, "b": 3}
    child = {"a": {"y": 99, "z": 7}}
    merged = _deep_merge(parent, child)
    assert merged == {"a": {"x": 1, "y": 99, "z": 7}, "b": 3}


# --------------------------------------------------------------------------- #
# Validation catches malformed inputs                                         #
# --------------------------------------------------------------------------- #


def test_tranche_attach_must_be_below_detach() -> None:
    with pytest.raises(ValueError, match="must be < detach"):
        TrancheConfig(name="bad", attach=0.6, detach=0.4)


def test_tranches_must_tile_zero_to_one(default_config: Config) -> None:
    payload = default_config.model_dump()
    payload["tranches"] = [
        {"name": "equity", "attach": 0.0, "detach": 0.25},
        {"name": "senior", "attach": 0.50, "detach": 1.0},  # gap [0.25, 0.50]
    ]
    with pytest.raises(ValueError, match="gap"):
        Config.model_validate(payload)


def test_tranches_must_start_at_zero(default_config: Config) -> None:
    payload = default_config.model_dump()
    payload["tranches"] = [
        {"name": "mezzanine", "attach": 0.10, "detach": 0.60},
        {"name": "senior", "attach": 0.60, "detach": 1.0},
    ]
    with pytest.raises(ValueError, match="start at attach=0"):
        Config.model_validate(payload)


def test_tranches_must_reach_one(default_config: Config) -> None:
    payload = default_config.model_dump()
    payload["tranches"] = [
        {"name": "equity", "attach": 0.0, "detach": 0.25},
        {"name": "senior", "attach": 0.25, "detach": 0.90},
    ]
    with pytest.raises(ValueError, match="cover up to detach=1"):
        Config.model_validate(payload)


def test_beta_lgd_infeasible_variance() -> None:
    with pytest.raises(ValueError, match="Beta with mean"):
        LgdConfig(distribution="beta", mean_init=0.85, std_init=0.40)


def test_default_model_must_be_in_models(default_config: Config) -> None:
    payload = default_config.model_dump()
    payload["credit"]["default_model"] = "cox_intensity"
    payload["credit"]["models"] = ["gaussian_copula", "student_t_copula"]
    with pytest.raises(ValueError, match="must be in models"):
        Config.model_validate(payload)
