"""``cash.cache(est.fit)`` keys the code ``fit`` reaches through ``self``.

A bound method carries its ``self`` as ``__self__``, not in the call's
arguments. The key looked for ``self`` in the arguments only, so what the
body reached through it -- a subclass's override of a method and the module
helpers that override calls -- was never folded. Editing such a helper kept
the key, and a later process was served the old result whenever the run-time
trace was off (``CASH_TRACE=0``, Python before 3.12) or could not be built
(here: a helper behind a closure decorator).

Each run is a fresh interpreter over one cache folder.
"""

from __future__ import annotations

import pytest

from tests.test_core.code_identity._edited_project import edited_runs

pytestmark = pytest.mark.core

HELPERS = """
def deco(fn):
    def wrapper(*a):
        return fn(*a)
    return wrapper


def plain(x):
    return x + 10


@deco
def wrapped(x):
    return x + 10


class Base:
    def __init__(self, k):
        self.k = k

    def fit(self):
        return self.null() + 1

    def null(self):
        return 0


class Sub(Base):
    def null(self):
        return HELPER(self.k * 10)
"""

CALLS = {
    "bound": "f = cash.cache(est.fit)\nprint(f())",
    "unbound": "f = cash.cache(helpers.Base.fit)\nprint(f(est))",
}


def _files(helper: str, call: str, trace: str = "1") -> dict[str, str]:
    # Set before cash is imported: the test runner keeps CASH_* out of a child.
    main = f"import os\nos.environ['CASH_TRACE'] = '{trace}'\nimport cash\nimport helpers\n\nest = helpers.Sub(1)\n{call}\n"
    return {"helpers.py": HELPERS.replace("HELPER", helper), "main.py": main}


@pytest.mark.parametrize("trace", ["1", "0"])
@pytest.mark.parametrize("helper", ["plain", "wrapped"])
@pytest.mark.parametrize("call", list(CALLS))
def test_editing_a_helper_an_override_calls_recomputes(tmp_path, call, helper, trace):
    edit = ("helpers.py", f"def {helper}(x):\n    return x + 10", f"def {helper}(x):\n    return x + 200")
    first, after, uncached = edited_runs(tmp_path, _files(helper, CALLS[call], trace), [edit])
    assert first == "21"
    assert uncached == "211"
    assert after == uncached, f"{call} method served the result from before its helper was edited"


@pytest.mark.parametrize("call", list(CALLS))
def test_unchanged_code_is_still_served(tmp_path, call):
    files = _files("wrapped", CALLS[call])
    files["main.py"] += 'print("HITS", f.cache_info()["hits"])\n'
    first, after, _ = edited_runs(tmp_path, files)
    assert first.split() == ["21", "HITS", "0"]
    assert after.split() == ["21", "HITS", "1"]


def test_a_bound_method_of_a_subclass_keys_the_override(tmp_path):
    """The override itself, reached only through ``self``, is keyed too."""
    files = _files("plain", CALLS["bound"], trace="0")
    edit = ("helpers.py", "return plain(self.k * 10)", "return plain(self.k * 30)")
    first, after, uncached = edited_runs(tmp_path, files, [edit])
    assert first == "21"
    assert after == uncached == "41"
