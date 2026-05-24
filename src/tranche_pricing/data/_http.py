"""Thin HTTP helper used by every data-acquisition module.

Each upstream (INSEE, Banque de France, ECB, FRED, Ken French Library) is hit
through :func:`fetch_bytes`, which adds a small retry loop, sane timeouts, an
identifying ``User-Agent`` and a uniform error type so the calling code can
fall back to the local CSV snapshot.
"""

from __future__ import annotations

import logging
import time
from typing import Final

import requests

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT: Final[float] = 30.0
DEFAULT_RETRIES: Final[int] = 3
DEFAULT_BACKOFF: Final[float] = 1.5
USER_AGENT: Final[str] = "tranche-pricing-paris/0.1 (https://github.com/dan-allouche; quant research)"


class UpstreamError(RuntimeError):
    """Raised when a data source cannot be reached or returns unusable content."""


def fetch_bytes(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: dict[str, str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    backoff: float = DEFAULT_BACKOFF,
) -> bytes:
    """Download a URL and return its raw bytes.

    Retries with exponential backoff on transient failures (5xx, connection
    errors, timeouts). On permanent failure, raises :class:`UpstreamError` so
    the caller can fall back to a CSV snapshot.
    """
    merged_headers = {"User-Agent": USER_AGENT, "Accept": "*/*"}
    if headers:
        merged_headers.update(headers)

    last_exc: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            response = requests.get(
                url,
                headers=merged_headers,
                params=params,
                timeout=timeout,
            )
            response.raise_for_status()
            return response.content
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and 400 <= status < 500 and status != 429:
                # Client errors (other than rate-limit) won't be solved by retries.
                raise UpstreamError(f"{url} returned HTTP {status}") from exc
            last_exc = exc
            wait = backoff**attempt
            logger.warning(
                "GET %s failed (attempt %d/%d): %s — retrying in %.1fs",
                url,
                attempt,
                retries,
                exc,
                wait,
            )
            time.sleep(wait)

    raise UpstreamError(f"GET {url} failed after {retries} attempts") from last_exc


__all__ = ["UpstreamError", "fetch_bytes"]
