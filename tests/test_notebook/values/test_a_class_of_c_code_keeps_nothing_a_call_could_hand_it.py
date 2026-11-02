"""A class written in C keeps nothing a call could hand it: its own
methods are descriptors of its code.

``from datetime import datetime`` binds a class whose ``now`` and
``strptime`` were taken for state of its own. After it, every statement
that called anything while reading a list too big to look through for the
classes it reaches (``print(len(logs))`` over 87,000 sessions) searched
the heap for what keeps the list, 2 to 4 s each, and warned
NOTEBOOK-SHARE-UNCHECKED. A class of the notebook's with a list of its own
still counts, when the call changed it.

A class whose state the call left as it was kept nothing new either: an
estimator's ``get_feature_names_out()``, on an object holding an array of
strings too big to read, linked it to every estimator class the notebook
imported, so the next in-place change of a list holding it moved every
pipeline and search on.
"""

from __future__ import annotations

import warnings

import pytest

from tests._cell_driver import run_cash_cell

BIG = "big = [[(str(j), j) for j in range(20)] for i in range(60_000)]"


def _unchecked(magics, setup: str, read: str = "print(len(big))") -> list[str]:
    run_cash_cell(magics, f"{setup}\n{BIG}")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(magics, read)
    return [str(w.message) for w in caught if "NOTEBOOK-SHARE-UNCHECKED" in str(w.message)]


@pytest.mark.parametrize(
    "setup",
    ["from datetime import datetime", "from collections import defaultdict"],
    ids=["datetime", "defaultdict"],
)
def test_a_c_class_leaves_a_read_of_a_big_list_unwarned(cash_magics, setup):
    assert _unchecked(cash_magics, setup) == []


REGISTRY = "class Registry:\n    held = []\ndef keep(x):\n    Registry.held.append(x)"


def test_a_class_with_a_list_of_its_own_still_counts(cash_magics):
    assert _unchecked(cash_magics, REGISTRY, "keep(big)") != []


def test_a_class_with_a_list_the_call_left_alone_is_no_holder(cash_magics):
    assert _unchecked(cash_magics, REGISTRY, "print(len(big))") == []
