"""A class module data holds is keyed by the methods a call ran, not all of them.

A cached function reads an object or a table of classes from a module of the
user's (``model.SVC["m"].fit(x)``, a library installed as an editable
checkout). Its key held each such class by all its code and all the code
that code names (``Model`` -> ``Minimizer`` and every method of each), so
editing ``Minimizer.shgo`` recomputed a fit that never calls it. While the
call's body is recorded (Python 3.12+, ``CASH_TRACE`` on) the key holds the
class by its surface -- its name, its bases, what it holds, its methods by
name -- and the trace checks the code of each method that ran, and what
that code read. With the trace off, or when it could not be recorded, the
class is keyed by all its code as before.

Each run is a fresh interpreter over one cache folder; every result is
compared with the same program run without cash.
"""

from __future__ import annotations

import sys

import pytest

from tests.test_core.code_identity._edited_project import edited_runs

pytestmark = pytest.mark.core

TRACED = sys.version_info >= (3, 12)

MINIMIZER = """
SCALE = 3
SHGO_TOL = 1


class Base:
    OFFSET = 100

    def from_base(self, x):
        return x + self.OFFSET

    def via_super(self, x):
        return x + 1000

    def base_unused(self, x):
        return x - 1


class Minimizer(Base):
    UNUSED = 1

    def __init__(self, k):
        self.k = k

    def minimize(self, x):
        return (
            self.leastsq(x)
            + self.from_base(0)
            + self.via_super(0)
            + self.scaled
            + self.factor()
            + self.halve(20)
        )

    def leastsq(self, x):
        return x * self.k * SCALE

    def via_super(self, x):
        return super().via_super(x)

    @property
    def scaled(self):
        return 10000

    @classmethod
    def factor(cls):
        return 100000

    @staticmethod
    def halve(v):
        return v // 2

    def shgo(self, x):
        return x - SHGO_TOL
"""

MODEL = """
from minimizer import Minimizer


class Model:
    def __init__(self):
        self.k = 2

    def fit(self, x):
        return self.run(x)

    def run(self, x):
        return Minimizer(self.k).minimize(x)

    def report(self):
        return Minimizer(1).shgo(0)


SVC = {"m": Model()}
CLASSES = {"model": Model}
"""

#: How the cached function reaches the held class.
REACH = {
    "instance": "import model\n\ndef reach():\n    return model.SVC['m']\n",
    "class_table": "import model\n\ndef reach():\n    return model.CLASSES['model']()\n",
    "global": "from model import SVC\n\ndef reach():\n    return SVC['m']\n",
}

MAIN = """
import os
os.environ["CASH_TRACE"] = "{trace}"
import cash
{reach}

@cash.cache
def fit(x):
    return {result}

print({shown}, fit.cache_info()["hits"])
"""

#: Ways a call's trace is left with nothing to check: code that ran which no
#: name holds (a lambda an instance made at import), and a result the body
#: produces as the caller reads it, after the call.
UNTRACED = {
    "unnamed code": (
        "self.run(x)",
        "self.post(self.run(x))",
        "        self.k = 2\n",
        "        self.k = 2\n        self.post = make_post(0)\n",
    ),
    "iterator": None,
}

#: Edits to code no call runs: served again, as plain Python computes the same.
UNRUN = {
    "uncalled method": ("minimizer.py", "return x - SHGO_TOL", "return x - SHGO_TOL - 5"),
    "uncalled method of the held class": ("model.py", "Minimizer(1).shgo(0)", "Minimizer(1).shgo(7)"),
    "uncalled base method": ("minimizer.py", "return x - 1", "return x - 9"),
    "constant only an uncalled method reads": ("minimizer.py", "SHGO_TOL = 1", "SHGO_TOL = 7"),
}

#: Edits to code a call runs, each reached its own way: recomputed.
RUN = {
    "called directly": ("minimizer.py", "x * self.k * SCALE", "x * self.k * SCALE * 2"),
    "via another method": ("model.py", "Minimizer(self.k).minimize(x)", "Minimizer(self.k).minimize(x) + 1"),
    "via a base class": ("minimizer.py", "return x + self.OFFSET", "return x + self.OFFSET + 3"),
    "via super()": ("minimizer.py", "return x + 1000", "return x + 2000"),
    "via a property": ("minimizer.py", "return 10000", "return 30000"),
    "via a classmethod": ("minimizer.py", "return 100000", "return 300000"),
    "via a staticmethod": ("minimizer.py", "return v // 2", "return v // 4"),
    "constant a called method reads": ("minimizer.py", "SCALE = 3", "SCALE = 5"),
    "class attribute a called method reads": ("minimizer.py", "OFFSET = 100", "OFFSET = 400"),
}

#: Edits to a class's surface: whatever the cache does, the result is plain's.
SURFACE = {
    "add a method": [
        (
            "minimizer.py",
            "    def shgo(self, x):",
            "    def brute(self, x):\n        return x\n\n    def shgo(self, x):",
        )
    ],
    "add an override that runs": [
        (
            "minimizer.py",
            "    def leastsq(self, x):",
            "    def from_base(self, x):\n        return x + 7\n\n    def leastsq(self, x):",
        )
    ],
    "remove a method": [("minimizer.py", "    def shgo(self, x):\n        return x - SHGO_TOL\n", "")],
    "rename an uncalled method": [("minimizer.py", "def shgo(self", "def shgo_renamed(self")],
    "rename a called method": [
        ("minimizer.py", "def halve(v)", "def halve_renamed(v)"),
        ("minimizer.py", "self.halve(20)", "self.halve_renamed(20)"),
    ],
    "rename a called method and its body": [
        ("minimizer.py", "def halve(v):\n        return v // 2", "def halve_renamed(v):\n        return v // 5"),
        ("minimizer.py", "self.halve(20)", "self.halve_renamed(20)"),
    ],
    "change an unread class attribute": [("minimizer.py", "UNUSED = 1", "UNUSED = 2")],
}

EXPECTED = "111122"


def _files(reach: str, trace: str = "1", untraced: str | None = None) -> dict[str, str]:
    result, shown, model = "reach().fit(x)", "fit(2)", MODEL
    if untraced == "iterator":
        result, shown = "iter([reach().fit(x)])", "list(fit(2))[0]"
    elif untraced is not None:
        call, call_after, init, init_after = UNTRACED[untraced]
        model = model.replace(call, call_after, 1).replace(init, init_after, 1)
        model = model.replace(
            "\n\nclass Model:", "\n\ndef make_post(k):\n    return lambda v: v + k\n\n\nclass Model:", 1
        )
    main = MAIN.format(trace=trace, reach=REACH[reach], result=result, shown=shown)
    return {"minimizer.py": MINIMIZER, "model.py": model, "main.py": main}


def _value_and_hits(out: str) -> tuple[str, int]:
    value, hits = out.split()
    return value, int(hits)


@pytest.mark.skipif(not TRACED, reason="the code trace needs sys.monitoring (Python 3.12+)")
@pytest.mark.parametrize("reach", list(REACH))
@pytest.mark.parametrize("edit", list(UNRUN))
def test_an_edit_to_code_no_call_runs_is_served(tmp_path, reach, edit):
    first, after, plain = edited_runs(tmp_path, _files(reach), [UNRUN[edit]])
    assert _value_and_hits(first) == (EXPECTED, 0)
    assert _value_and_hits(plain)[0] == EXPECTED
    assert _value_and_hits(after) == (EXPECTED, 1), f"{edit}: recomputed though no call runs what changed"


@pytest.mark.parametrize("trace", ["1", "0"])
@pytest.mark.parametrize("reach", list(REACH))
@pytest.mark.parametrize("edit", list(RUN))
def test_an_edit_to_code_a_call_runs_recomputes(tmp_path, reach, edit, trace):
    first, after, plain = edited_runs(tmp_path, _files(reach, trace), [RUN[edit]])
    assert _value_and_hits(first) == (EXPECTED, 0)
    assert _value_and_hits(plain)[0] != EXPECTED
    assert _value_and_hits(after) == (_value_and_hits(plain)[0], 0), f"{edit}: served the result from before"


@pytest.mark.parametrize("reach", ["instance", "class_table"])
@pytest.mark.parametrize("edit", list(SURFACE))
def test_an_edit_to_a_class_s_surface_computes_what_plain_python_does(tmp_path, reach, edit):
    first, after, plain = edited_runs(tmp_path, _files(reach), SURFACE[edit])
    assert _value_and_hits(first) == (EXPECTED, 0)
    assert _value_and_hits(after)[0] == _value_and_hits(plain)[0], edit


@pytest.mark.parametrize("reach", list(REACH))
@pytest.mark.parametrize("edit", list(UNRUN))
def test_with_the_trace_off_an_edit_to_code_no_call_runs_still_recomputes(tmp_path, reach, edit):
    first, after, plain = edited_runs(tmp_path, _files(reach, trace="0"), [UNRUN[edit]])
    assert _value_and_hits(first) == (EXPECTED, 0)
    assert _value_and_hits(after) == (_value_and_hits(plain)[0], 0)


@pytest.mark.parametrize("reach", list(REACH))
def test_unchanged_code_is_served_with_the_trace_on_or_off(tmp_path, reach):
    for trace in ("1", "0"):
        (tmp_path / trace).mkdir()
        first, after, plain = edited_runs(tmp_path / trace, _files(reach, trace), [])
        assert _value_and_hits(first) == (EXPECTED, 0)
        assert _value_and_hits(after) == (EXPECTED, 1)
        assert _value_and_hits(plain)[0] == EXPECTED


@pytest.mark.skipif(not TRACED, reason="the code trace needs sys.monitoring (Python 3.12+)")
@pytest.mark.parametrize("untraced", list(UNTRACED))
@pytest.mark.parametrize("edit", ["unchanged", "uncalled method", "called directly"])
def test_a_call_whose_trace_cannot_be_built_is_checked_on_all_the_code(tmp_path, edit, untraced):
    """A call whose trace has nothing to check is stored with a check of
    every held class's code in full instead: an edit anywhere in them
    recomputes, as with the trace off, and nothing is served stale."""
    edits = [] if edit == "unchanged" else [{**UNRUN, **RUN}[edit]]
    first, after, plain = edited_runs(tmp_path, _files("instance", untraced=untraced), edits)
    assert _value_and_hits(first) == (EXPECTED, 0)
    expected_hits = 1 if edit == "unchanged" else 0
    assert _value_and_hits(after) == (_value_and_hits(plain)[0], expected_hits)
