"""External data acquisition.

Each submodule exposes one ``fetch`` function returning a tidy
``pandas.DataFrame`` with a ``date`` column and one or more series columns.
The functions all share the same signature pattern::

    df = fetch(start=..., end=..., refresh=False)

When ``refresh`` is False (default), the local snapshot under ``data/raw/``
is read directly so the pipeline runs offline. When ``refresh`` is True (or
when no snapshot exists yet), the live upstream is hit and the snapshot is
rewritten with the new payload.
"""

from __future__ import annotations

from . import (
    cache,
    ecb,
    fama_french,
    fred,
    insee_irl,
    insee_unemployment,
    notaires,
    oat,
    visale,
)

# Convenience map from canonical series name to its `fetch` callable. The CLI
# and the notebooks iterate over this dict to refresh every source in one go.
SERIES = {
    notaires.SERIES_NAME: notaires.fetch,
    insee_irl.SERIES_NAME: insee_irl.fetch,
    insee_unemployment.SERIES_NAME: insee_unemployment.fetch,
    oat.SERIES_NAME: oat.fetch,
    ecb.SERIES_NAME: ecb.fetch,
    fred.SERIES_NAME: fred.fetch,
    fama_french.SERIES_NAME: fama_french.fetch,
    visale.SERIES_NAME: visale.fetch,
}

__all__ = [
    "SERIES",
    "cache",
    "ecb",
    "fama_french",
    "fred",
    "insee_irl",
    "insee_unemployment",
    "notaires",
    "oat",
    "visale",
]
