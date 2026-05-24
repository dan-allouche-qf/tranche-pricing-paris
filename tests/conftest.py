"""Shared pytest fixtures.

Fixtures defined here are auto-discovered by every test module. We keep this
file deliberately small: heavy data / Monte-Carlo fixtures live next to the
tests that use them.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from tranche_pricing.config import Config, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "config"


@pytest.fixture(scope="session")
def repo_root() -> Path:
    """Absolute path to the repository root."""
    return REPO_ROOT


@pytest.fixture(scope="session")
def config_dir() -> Path:
    """Absolute path to the config/ directory."""
    return CONFIG_DIR


@pytest.fixture(scope="session")
def default_config() -> Config:
    """The default toy-building configuration."""
    return load_config(CONFIG_DIR / "default.yaml")


@pytest.fixture(scope="session")
def paris_config() -> Config:
    """The realistic Paris configuration (70 apartments)."""
    return load_config(CONFIG_DIR / "paris_intermediate.yaml")
