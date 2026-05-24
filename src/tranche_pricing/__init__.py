"""Tranche pricing on Paris residential rental cash flows."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("tranche-pricing-paris")
except PackageNotFoundError:  # pragma: no cover - editable install pre-build
    __version__ = "0.0.0"

__all__ = ["__version__"]
