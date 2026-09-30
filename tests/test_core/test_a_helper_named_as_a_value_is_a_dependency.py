"""A helper named as a value through its module -- ``map(helper.g, xs)``,
``fn = helper.g``, ``df.apply(helper.g)`` -- is a dependency.

Only a helper CALLED as ``helper.g(x)``, or named bare after ``from helper
import g``, was followed; handed on by its module attribute, an edit to it
served the old result with no warning.
"""

from __future__ import annotations

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S
from tests.test_core._edited_project import edited_runs

pytestmark = pytest.mark.core

HELPER = """
def g(x):
    return x + 1


class C:
    def __init__(self, x):
        self.v = x + 1

    @staticmethod
    def m(x):
        return x + 1
"""

EDIT_G = ("def g(x):\n    return x + 1", "def g(x):\n    return x + 100")
EDIT_C = ("self.v = x + 1", "self.v = x + 100")
EDIT_M = ("def m(x):\n        return x + 1", "def m(x):\n        return x + 100")

BODIES = {
    "map": ("return list(map(helper.g, [x]))[0]", EDIT_G),
    "bound to a local": ("fn = helper.g\n    return fn(x)", EDIT_G),
    "passed to a plain function": ("return apply(helper.g, x)", EDIT_G),
    "under a partial": ("return functools.partial(helper.g)(x)", EDIT_G),
    "to an executor": (
        "with concurrent.futures.ThreadPoolExecutor(1) as ex:\n        return list(ex.map(helper.g, [x]))[0]",
        EDIT_G,
    ),
    "a class": ("return list(map(helper.C, [x]))[0].v", EDIT_C),
    "a staticmethod": ("return list(map(helper.C.m, [x]))[0]", EDIT_M),
}


@pytest.mark.parametrize("body", list(BODIES))
def test_editing_a_helper_named_as_a_value_recomputes(tmp_path, body):
    code, edit = BODIES[body]
    main = f"""
import concurrent.futures
import functools
import time

import cash
import helper


def apply(fn, x):
    return fn(x)


@cash.cache
def f(x):
    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})
    {code}


print(f(1))
"""
    first, after, uncached = edited_runs(tmp_path, {"main.py": main, "helper.py": HELPER}, [("helper.py", *edit)])
    assert first == "2"
    assert after == uncached == "101"
