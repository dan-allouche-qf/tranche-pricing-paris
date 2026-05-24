"""Persistent cache layer for external data series.

Every ``fetch(start, end)`` call in :mod:`tranche_pricing.data` is memoised here
so that re-running ``make data`` is essentially free as long as the underlying
upstream signature does not change. We layer two mechanisms:

* a ``joblib.Memory`` cache that stores the raw parsed payload under
  ``data/.cache/<namespace>/``;
* a JSON provenance log at ``data/raw/_provenance.json`` that records the
  retrieval timestamp, source URL and content hash for every successful pull.

Provenance is what gets shown in the working paper's data appendix.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Callable
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
from joblib import Memory

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / ".cache"
PROVENANCE_PATH = RAW_DIR / "_provenance.json"

# Make sure the expected directory layout exists. Cheap and idempotent.
for _d in (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

_memory = Memory(location=str(CACHE_DIR), verbose=0)


F = TypeVar("F", bound=Callable[..., pd.DataFrame])


def cached(func: F) -> F:
    """Decorator that memoises a fetch function on (args, kwargs).

    The function MUST return a ``pandas.DataFrame``. Cache invalidation happens
    naturally when the function source changes (joblib hashes the bytecode).
    """
    memoised = _memory.cache(func)

    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> pd.DataFrame:
        df = memoised(*args, **kwargs)
        # Always return a fresh copy so downstream mutations cannot poison
        # the in-memory cache layer that joblib keeps for the session.
        return df.copy()

    return wrapper  # type: ignore[return-value]


# --------------------------------------------------------------------------- #
# Provenance                                                                  #
# --------------------------------------------------------------------------- #


def _load_provenance() -> dict[str, dict[str, Any]]:
    if not PROVENANCE_PATH.exists():
        return {}
    try:
        with PROVENANCE_PATH.open() as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        logger.warning("Provenance file at %s is corrupt; starting fresh.", PROVENANCE_PATH)
        return {}


def _save_provenance(records: dict[str, dict[str, Any]]) -> None:
    PROVENANCE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PROVENANCE_PATH.open("w") as fh:
        json.dump(records, fh, indent=2, sort_keys=True)


def record_provenance(
    name: str,
    *,
    url: str,
    rows: int,
    start: str | None,
    end: str | None,
    extra: dict[str, Any] | None = None,
) -> None:
    """Append/update a provenance entry for one external series."""
    records = _load_provenance()
    records[name] = {
        "retrieved_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "url": url,
        "rows": int(rows),
        "start": start,
        "end": end,
        **(extra or {}),
    }
    _save_provenance(records)
    logger.info("Provenance recorded for %s (%d rows).", name, rows)


def get_provenance(name: str) -> dict[str, Any] | None:
    return _load_provenance().get(name)


# --------------------------------------------------------------------------- #
# CSV snapshot fallback helpers                                               #
# --------------------------------------------------------------------------- #


def hash_dataframe(df: pd.DataFrame) -> str:
    """Stable content hash, used in provenance to detect upstream changes."""
    h = hashlib.sha256()
    h.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return h.hexdigest()[:16]


def read_snapshot(name: str) -> pd.DataFrame:
    """Load the local CSV snapshot of a series from ``data/raw/<name>.csv``.

    Raises ``FileNotFoundError`` if the snapshot has not been committed yet.
    The expected schema is ``date,value`` with one row per observation.
    """
    path = RAW_DIR / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No local snapshot for {name!r} at {path}. "
            "Either run `make data` to refresh, or drop a CSV with columns "
            "date,value into data/raw/."
        )
    df = pd.read_csv(path, parse_dates=["date"])
    return df.sort_values("date").reset_index(drop=True)


def write_snapshot(name: str, df: pd.DataFrame) -> Path:
    """Persist a fetched series under ``data/raw/<name>.csv``."""
    path = RAW_DIR / f"{name}.csv"
    df.sort_values("date").reset_index(drop=True).to_csv(path, index=False)
    return path


def clear_cache() -> None:
    """Drop every memoised result. Useful from tests and notebooks."""
    _memory.clear(warn=False)


__all__ = [
    "CACHE_DIR",
    "DATA_DIR",
    "INTERIM_DIR",
    "PROCESSED_DIR",
    "PROVENANCE_PATH",
    "RAW_DIR",
    "cached",
    "clear_cache",
    "get_provenance",
    "hash_dataframe",
    "read_snapshot",
    "record_provenance",
    "write_snapshot",
]
