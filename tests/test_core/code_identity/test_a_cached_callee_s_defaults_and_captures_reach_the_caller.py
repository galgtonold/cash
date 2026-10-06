"""What a cached callee is bound to reaches its cached caller's key.

A cached function that calls another cached function keyed the callee by its
code. The callee's parameter defaults and the variables it captures live on
the function object, not in its code: editing the constant a default comes
from, or building the callee with another captured value, served the
caller's old result while the callee called directly recomputed.
"""

from __future__ import annotations

from tests.test_core.code_identity._edited_project import edited_runs

DEFAULT_MAIN = """
    import cash
    from lib import clip

    @cash.cache
    def report(xs):
        return [clip(x) for x in xs]

    print(report([1, 10, 100]))
"""

DEFAULT_LIB = """
    import cash
    from settings import THRESHOLD

    @cash.cache
    def clip(x, limit=THRESHOLD):
        return min(x, limit)
"""


def test_a_default_from_an_edited_constant_reaches_the_caller(tmp_path):
    first, after, uncached = edited_runs(
        tmp_path,
        {"main.py": DEFAULT_MAIN, "lib.py": DEFAULT_LIB, "settings.py": "THRESHOLD = 5\n"},
        [("settings.py", "THRESHOLD = 5", "THRESHOLD = 50")],
    )
    assert first == "[1, 5, 5]"
    assert after == uncached == "[1, 10, 50]"


CLOSURE_MAIN = """
    import os
    import cash

    def make(n):
        @cash.cache
        def inner(x):
            return x + n
        return inner

    add = make(int(os.environ["STEP"]))

    @cash.cache
    def total(x):
        return add(x)

    print(total(0))
"""


def test_a_captured_value_of_a_factory_made_callee_reaches_the_caller(tmp_path, monkeypatch):
    monkeypatch.setenv("STEP", "5")
    first, after, uncached = edited_runs(tmp_path, {"main.py": CLOSURE_MAIN}, env_after={"STEP": "50"})
    assert first == "5"
    assert after == uncached == "50"
