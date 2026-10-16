"""``Ridge().fit(X, y)`` in a notebook is not an unseeded fit.

Ridge uses its ``random_state`` only with ``solver='sag'`` or ``'saga'``, so
with the default solver its fit is deterministic; the notebook still warned
"Unseeded randomness detected ... two genuine fits would differ" for it, both
for an inline fit and for a bare fit's receiver. A sag fit still
warns.
"""

from __future__ import annotations

import warnings

import pytest

from cash.tracking.randomness import CashRandomnessWarning
from tests._cell_driver import run_cash_cell

pytest.importorskip("sklearn")

SETUP = [
    "import numpy as np\nfrom sklearn.linear_model import Ridge",
    "X = np.random.RandomState(0).rand(50, 3)\ny = X @ np.array([1.0, 2.0, 3.0])",
]


def _messages(magics, cells: list[str]) -> list[str]:
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for cell in [*SETUP, *cells]:
            run_cash_cell(magics, cell)
    return [str(w.message) for w in caught if issubclass(w.category, CashRandomnessWarning)]


def test_an_inline_ridge_fit_is_silent(cash_magics):
    assert not _messages(cash_magics, ["m = Ridge(alpha=0.5).fit(X, y)"])


def test_a_cache_fit_ridge_receiver_is_silent(cash_magics):
    assert not _messages(cash_magics, ["m = Ridge()", "m.fit(X, y)"])


def test_a_sag_fit_still_warns(cash_magics):
    messages = _messages(cash_magics, ["m = Ridge(solver='sag').fit(X, y)"])
    assert any("Unseeded randomness detected" in m for m in messages), messages
