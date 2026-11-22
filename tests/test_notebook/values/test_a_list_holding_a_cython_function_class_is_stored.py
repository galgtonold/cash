"""A value holding the class of a Cython function is checked like any other.

Cython's function classes (``cython_function_or_method``) are made by its
``_common_types_metatype``, which has no ``__module__`` of its own: read on
the metatype, it gives ``type``'s descriptor, not a string. The check for
outputs that cannot be restored read it as a module name and raised, and the
statement failed with ``'getset_descriptor' object has no attribute
'startswith'``, though its own code was fine: once matplotlib.figure was loaded,
for any container holding such a class next to an object.
"""

from __future__ import annotations

import pytest

from cash.value_types import class_module, mro_kind
from tests._cell_driver import run_cash_cell

mh = pytest.importorskip("sklearn.utils.murmurhash")
pytest.importorskip("matplotlib")


def test_the_metatype_has_no_module_name():
    kind = type(mh.murmurhash3_32)
    assert class_module(type(kind)) == ""
    assert mro_kind(kind, {"matplotlib.figure.Figure": "figure"}, ("matplotlib",)) is None


def test_a_statement_keeping_the_class_runs_and_is_stored(cash_magics):
    # matplotlib.figure loaded: the check looks inside containers for figures.
    run_cash_cell(cash_magics, "import matplotlib.figure\nimport sklearn.utils.murmurhash as mh")
    run_cash_cell(cash_magics, "kinds = [type(mh.murmurhash3_32), object()]\nn = len(kinds)")
    assert cash_magics.shell.user_ns["n"] == 2
    row = cash_magics._last_cell_metrics["statements"][-1]
    assert str(row["status"]) == "COMPUTED"
