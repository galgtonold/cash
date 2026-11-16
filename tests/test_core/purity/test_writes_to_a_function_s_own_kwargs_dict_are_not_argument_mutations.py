"""A write to a function's own ``**kwargs`` dict changes nothing the caller holds.

CPython builds the ``**kwargs`` dict anew on every call, even for
``f(**d)``, so ``kwargs.update(...)``, ``kwargs[k] = v``, ``kwargs.pop(k)``
and ``del kwargs[k]`` change only the call's own dict. They were reported as
mutations of the argument ``kwargs``, which refused every traced call of a
library whose constructors normalise their keyword options so (lmfit's
``Model.__init__``). A write INTO a value the dict holds is the caller's
object and is still reported.
"""

from __future__ import annotations

import pytest

import cash
from cash.analysis.purity_analyzer import PurityAnalyzer

pytestmark = pytest.mark.core


def _issues(fn):
    return [(i.kind, i.description) for i in PurityAnalyzer().analyze(fn).issues]


def normalised(**kwargs):
    kwargs.update(prefix="")
    kwargs["nan_policy"] = "raise"
    kwargs.pop("unused", None)
    kwargs.setdefault("scale", 1)
    del kwargs["nan_policy"]
    kwargs |= {"fixed": True}
    return sorted(kwargs)


class Model:
    def __init__(self, func, **kws):
        kws.update({"prefix": ""})
        self.func = func
        self.opts = kws


def into_a_held_value(**kw):
    kw["rows"].append(1)
    return len(kw["rows"])


def into_a_value_taken_out(**kw):
    rows = kw["rows"]
    rows.append(1)
    return len(rows)


@pytest.mark.parametrize("fn", [normalised, Model.__init__], ids=["function", "constructor"])
def test_writes_to_the_own_kwargs_dict_are_not_reported(fn):
    assert not [d for _, d in _issues(fn) if "'kwargs'" in d or "'kws'" in d]


@pytest.mark.parametrize("fn", [into_a_held_value, into_a_value_taken_out], ids=["subscript", "name"])
def test_a_write_into_a_value_the_dict_holds_is_still_reported(fn):
    assert _issues(fn), "a write into the caller's list went unreported"


def test_a_cached_function_writing_its_kwargs_is_cached_and_leaves_the_caller_s_dict(tmp_path):
    c = cash.Cash(cache_dir=str(tmp_path))
    cached = c.cache(normalised)
    options = {"unused": 1, "a": 2}
    plain = normalised(**options)
    assert cached(**options) == plain
    assert cached(**options) == plain
    assert options == {"unused": 1, "a": 2}
    assert cached.cache_info()["hits"] == 1


def test_a_cached_function_writing_into_a_caller_s_list_changes_it_as_plain_python_does(tmp_path):
    c = cash.Cash(cache_dir=str(tmp_path))
    cached = c.cache(into_a_held_value)
    mine, theirs = [], []
    for _ in range(2):
        assert cached(rows=mine) == into_a_held_value(rows=theirs)
    assert mine == theirs == [1, 1]
    assert cached.cache_info()["hits"] == 0
