"""A statement whose draw happens inside a function it calls.

``x = make_data()``, where ``make_data`` lives in a module and draws from
numpy's global stream, shows no draw in the statement's source. The
before/after comparison of the streams saw it and marked the badge, but no
RANDOM-UNSEEDED was given, so the frozen value went unremarked in the output.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from cash.tracking.randomness import CashRandomnessWarning, lineage
from tests._cell_driver import run_cash_cell


@pytest.fixture(autouse=True)
def nothing_seeded(monkeypatch):
    monkeypatch.setattr(lineage, "_PROCESS_SEEDED", set())


def make_data():
    return np.random.rand(3).tolist()


def make_seeded_data():
    np.random.seed(0)
    return np.random.rand(3).tolist()


def mean_of(n):
    return float(np.arange(n).mean())


_IMPORT = f"import numpy as np\nfrom {__name__} import make_data, make_seeded_data, mean_of"


def _messages(magics, cells):
    cells = [_IMPORT, *cells]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for cell in cells:
            run_cash_cell(magics, cell, cells=cells)
    return [str(w.message) for w in caught if issubclass(w.category, CashRandomnessWarning)]


def test_a_draw_inside_a_called_function_warns(full_warning_text, cash_magics):
    found = _messages(cash_magics, ["x = make_data()"])
    assert len(found) == 1, found
    assert "RANDOM-UNSEEDED" in found[0]
    assert "drew from numpy's global random stream without a seed" in found[0]


@pytest.mark.parametrize(
    "cells",
    [
        ["x = make_seeded_data()"],
        ["np.random.seed(1)", "x = make_data()"],
        ["x = mean_of(5)"],
        ["# @cash:allow-random\nx = make_data()"],
    ],
    ids=["seeds-itself", "seeded-above", "no-draw", "allow-random"],
)
def test_a_seeded_or_silenced_draw_does_not(full_warning_text, cash_magics, cells):
    assert not _messages(cash_magics, cells)
