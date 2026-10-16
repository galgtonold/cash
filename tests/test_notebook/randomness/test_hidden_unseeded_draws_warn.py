"""Draws the source scan could not attribute to a generator.

``df.sample(100)`` (no ``random_state``), ``np.random.default_rng().normal()``
and ``rng = make_rng()``, where ``make_rng`` builds a generator with no seed,
move no global stream, so neither the scan nor the before/after stream check
saw them: the first result was cached and replayed, frozen, without a word.
Each now gives RANDOM-UNSEEDED; the seeded spellings stay silent.
"""

from __future__ import annotations

import warnings

import pytest

from cash.tracking.randomness import CashRandomnessWarning, lineage
from tests._cell_driver import run_cash_cell

pd = pytest.importorskip("pandas")


@pytest.fixture(autouse=True)
def nothing_seeded(monkeypatch):
    monkeypatch.setattr(lineage, "_PROCESS_SEEDED", set())


_SETUP = "import numpy as np\nimport pandas as pd\ndf = pd.DataFrame({'a': range(1000)})"
_HELPER = "def make_rng():\n    return np.random.default_rng()"
_SEEDED_HELPER = "def make_rng():\n    return np.random.default_rng(0)"


def _messages(magics, cells):
    cells = [_SETUP, *cells]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for cell in cells:
            run_cash_cell(magics, cell, cells=cells)
    return [str(w.message) for w in caught if issubclass(w.category, CashRandomnessWarning)]


@pytest.mark.parametrize(
    ("cells", "names"),
    [
        (["df.sample(100)"], "sample"),
        (["s = df.sample(frac=0.1)"], "sample"),
        (["z = np.random.default_rng().normal()"], "default_rng"),
        ([_HELPER, "rng = make_rng()"], "make_rng()"),
    ],
    ids=["bare-sample", "assigned-sample", "anonymous-generator", "helper-generator"],
)
def test_a_hidden_unseeded_draw_warns(full_warning_text, cash_magics, cells, names):
    found = [m for m in _messages(cash_magics, cells) if "RANDOM-UNSEEDED" in m]
    assert len(found) == 1, found
    assert names in found[0]


@pytest.mark.parametrize(
    "cells",
    [
        ["df.sample(100, random_state=0)"],
        ["z = np.random.default_rng(0).normal()"],
        [_SEEDED_HELPER, "rng = make_rng()"],
        ["# @cash:allow-random\ndf.sample(100)"],
        ["np.random.seed(0)", "s = df.sample(100)"],
    ],
    ids=["seeded-sample", "seeded-generator", "seeded-helper", "allow-random", "numpy-stream-seeded"],
)
def test_a_seeded_or_allowed_draw_stays_silent(full_warning_text, cash_magics, cells):
    found = [m for m in _messages(cash_magics, cells) if "RANDOM-UNSEEDED" in m]
    assert found == []
