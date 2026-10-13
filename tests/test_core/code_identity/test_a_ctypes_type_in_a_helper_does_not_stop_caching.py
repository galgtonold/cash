"""A call to a ctypes type does not stop the helper walk.

The walk asks of each callee whether it is a class built by ``exec()``, and
for a class looked through the classes nested in it. ``ctypes.c_int`` holds
its byte-swapped twin as ``__ctype_be__``, whose ``__ctype_le__`` is
``c_int`` again: the look went round that cycle until RecursionError, and
any cached function that called ``ctypes.c_int(...)`` -- LightGBM's
``fit`` does -- ran uncached with KEY-HELPERS-UNWALKABLE.
"""

from __future__ import annotations

import ctypes
import warnings

import pytest

from cash import Cash
from cash.exceptions import CashWarning

pytestmark = [pytest.mark.core]


def _as_c_int(x):
    return ctypes.c_int(x).value + 1


def test_a_cached_function_calling_a_ctypes_type_is_cached(tmp_path):
    c = Cash(cache_dir=str(tmp_path / "cache"), register_magic=False)
    f = c.cache(_as_c_int)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert f(3) == 4
        assert f(3) == 4
    unwalkable = [
        str(w.message) for w in caught if issubclass(w.category, CashWarning) and "UNWALKABLE" in str(w.message)
    ]
    assert not unwalkable, unwalkable
    assert f.cache_info()["hits"] == 1


def test_classes_that_hold_each_other_are_looked_through_once():
    from cash.analysis.helper_code import is_exec_built

    class A:
        pass

    class B:
        pass

    A.other, B.other = B, A
    A.itself = A
    assert is_exec_built(A) is False
    assert is_exec_built(ctypes.c_int) is False
