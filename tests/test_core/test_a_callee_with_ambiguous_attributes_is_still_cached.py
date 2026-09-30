"""A callee whose attributes refuse to be true or false is still keyed.

``pl.col`` answers any attribute with an expression (``pl.col.x`` is the
column ``x``), and an expression raises when asked whether it is true. The
helper walk asked ``if getattr(callee, "_cash_cached", False)``, which raised;
the failure was once taken for "no helpers" and later for "cannot key", so a
function calling ``pl.col(...)`` ran uncached. Only ``True`` itself marks a
cached function.
"""

from __future__ import annotations

import warnings

from cash import Cash


class _Ambiguous:
    def __bool__(self):
        raise TypeError("the truth value of an expression is ambiguous")


class _Columns:
    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Ambiguous()

    def __call__(self, value):
        return value * 2


col = _Columns()


def _double(x):
    return col(x)


def test_a_function_calling_it_is_cached_without_a_warning():
    c = Cash()
    double = c.cache(_double)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert double(3) == 6
        assert double(3) == 6
    codes = [getattr(w.message, "code", None) for w in caught]
    assert "KEY-HELPERS-UNWALKABLE" not in codes
    assert [call["cache_hit"] for call in c.drain_decorator_calls()] == [False, True]
