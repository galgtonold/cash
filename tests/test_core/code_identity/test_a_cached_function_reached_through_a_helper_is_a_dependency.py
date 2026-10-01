"""A cached function reached through a plain helper, a static or class method,
a decorated helper or an import inside the body is a dependency of its caller.

Only the calls written in the cached function's own body used to link it to
the cached functions it reaches. ``report()`` calling a plain ``prepare()``
that calls cached ``load()`` -- the natural shape of a pipeline -- kept its
old result after ``load``'s body, a helper of ``load``, a global it reads or
an environment variable it reads changed, and refreshed on its own TTL only.
"""

from __future__ import annotations

import time

import pytest

import cash
from tests.test_core._edited_project import edited_runs

pytestmark = pytest.mark.core

PIPELINE = """
import functools
import os
import time

import cash

G = 2


def scale(n):
    return n * G


@cash.cache
def load(n):
    return scale(n) + int(os.environ.get("BUMP", "0"))


{shape}


@cash.cache
def report(n):
    return {call}


print(report(2))
"""

SHAPES = {
    "plain helper": ("def prepare(n):\n    return load(n) + 1", "prepare(n)"),
    "staticmethod": ("class K:\n    @staticmethod\n    def prepare(n):\n        return load(n) + 1", "K.prepare(n)"),
    "classmethod": ("class K:\n    @classmethod\n    def prepare(cls, n):\n        return load(n) + 1", "K.prepare(n)"),
    "decorated helper": (
        "def traced(f):\n    @functools.wraps(f)\n    def inner(*a):\n        return f(*a)\n    return inner\n\n\n"
        "@traced\ndef prepare(n):\n    return load(n) + 1",
        "prepare(n)",
    ),
}

EDITS = {
    "its body": ([("main.py", "return scale(n) +", "return scale(n) * 10 +")], None),
    "its helper": ([("main.py", "return n * G", "return n * G * 7")], None),
    "a global it reads": ([("main.py", "G = 2", "G = 5")], None),
    "a variable it reads": ([], {"BUMP": "100"}),
}


@pytest.mark.parametrize("edit", list(EDITS))
@pytest.mark.parametrize("shape", list(SHAPES))
def test_editing_the_cached_function_recomputes_the_caller(tmp_path, shape, edit):
    helper, call = SHAPES[shape]
    edits, env = EDITS[edit]
    source = PIPELINE.format(shape=helper, call=call)
    first, after, uncached = edited_runs(tmp_path, {"main.py": source}, edits, env)
    assert first == "5"
    assert after == uncached != first


INNER = """
import cash


@cash.cache
def load(n):
    return n * 2
"""


@pytest.mark.parametrize(
    "body",
    ["from inner import load\n    return load(n) + 1", "import inner\n    return inner.load(n) + 1"],
    ids=["from-import", "import-module"],
)
def test_a_cached_function_imported_in_the_body_is_a_dependency(tmp_path, body):
    top = f"""
import time

import cash


@cash.cache
def report(n):
    {body}
"""
    files = {"inner.py": INNER, "top.py": top, "main.py": "import top\nprint(top.report(2))\n"}
    first, after, uncached = edited_runs(tmp_path, files, [("inner.py", "n * 2", "n * 30")])
    assert first == "5"
    assert after == uncached == "61"


def test_the_caller_refreshes_at_the_ttl_of_a_cached_function_behind_a_helper(tmp_path):
    """A result built from a TTL'd function is refreshed when that TTL runs out,
    as a direct call already made it."""
    c = cash.Cash(cache_dir=str(tmp_path / "c"))

    @c.cache(ttl=1, assume_safe=True)
    def price(n):
        return time.time()

    def prepare(n):
        return price(n)

    @c.cache
    def report(n):
        return prepare(n)

    first = report(1)
    assert report(1) == first, "the control failed: a second call should hit"
    time.sleep(1.2)  # let the 1 s TTL run out
    assert report(1) != first


def test_a_cached_clock_read_is_not_blamed_on_other_cached_functions(tmp_path):
    """Linking cached callees through helpers must not make every cached
    callee look like the clock read one of them returns (they share cash's
    wrapper code)."""
    import warnings

    from cash.exceptions import CashImpurityWarning

    c = cash.Cash(cache_dir=str(tmp_path / "c"))

    @c.cache(assume_safe=True)
    def stamp(n):
        return time.time()

    @c.cache
    def uses_stamp(n):
        return stamp(n)

    uses_stamp(1)

    @c.cache
    def load(n):
        return n * 2

    @c.cache
    def top(n):
        return load(n) + 1

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert top(1) == 3
    ambient = [str(w.message) for w in caught if issubclass(w.category, CashImpurityWarning)]
    assert not ambient, ambient
