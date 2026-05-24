"""Tests for ``_reconstruct_tranches``.

The helper walks the waterfall's ``notional_path`` dict (which the
waterfall populates in junior-to-senior attach order) and rebuilds the
``Tranche`` stack independently of the tranche names.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from tranche_pricing.pricing.tranche_pricer import _reconstruct_tranches


def _stub_output(notional_path: dict[str, np.ndarray], par: float) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(par=par),
        waterfall=SimpleNamespace(notional_path=notional_path),
    )


def test_reconstructs_non_canonical_names() -> None:
    par = 1_000.0
    # Stack: A=20%, B=30%, C=20%, D=30% of par (junior to senior).
    notionals = {
        "A": np.array([[200.0]]),
        "B": np.array([[300.0]]),
        "C": np.array([[200.0]]),
        "D": np.array([[300.0]]),
    }
    out = _stub_output(notionals, par=par)
    rebuilt = _reconstruct_tranches(out)
    assert [t.name for t in rebuilt] == ["A", "B", "C", "D"]
    assert rebuilt[0].attach == pytest.approx(0.0)
    assert rebuilt[0].detach == pytest.approx(0.2)
    assert rebuilt[1].attach == pytest.approx(0.2)
    assert rebuilt[1].detach == pytest.approx(0.5)
    assert rebuilt[2].attach == pytest.approx(0.5)
    assert rebuilt[2].detach == pytest.approx(0.7)
    assert rebuilt[3].attach == pytest.approx(0.7)
    assert rebuilt[3].detach == pytest.approx(1.0)


def test_reconstructs_canonical_stack_unchanged() -> None:
    par = 100.0
    # equity 15%, mezzanine 25%, senior 60% (junior to senior).
    notionals = {
        "equity": np.array([[15.0]]),
        "mezzanine": np.array([[25.0]]),
        "senior": np.array([[60.0]]),
    }
    out = _stub_output(notionals, par=par)
    rebuilt = _reconstruct_tranches(out)
    assert [t.name for t in rebuilt] == ["equity", "mezzanine", "senior"]
    assert rebuilt[-1].detach == pytest.approx(1.0)
