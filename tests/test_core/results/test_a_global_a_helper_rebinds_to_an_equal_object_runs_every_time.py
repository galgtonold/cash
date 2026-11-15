"""A call that binds an attribute of a module global to an equal NEW object,
through a helper one or two calls below a method of an argument's class, is
not stored: a hit would leave the old object where plain Python leaves the
new one.

``def _remember(holder, n): holder.last = [n]`` called as
``_remember(HELD, n)``, with ``HELD.last`` already an equal ``[n]``: the
drift guard compared the global's value digest before and after the call,
saw no change, and stored the entry, so the next hit skipped the write. What
each watched global's attributes are bound to is now snapshotted before the
body and compared by identity after it (`PurityChecks.rebound_globals`); a
global seen rebound so is snapshotted on every later call. A write that
leaves the same object (a small int) is no change, and a cached driver whose
helper writes only the value still runs when the value differs.
"""

from __future__ import annotations

import json
import textwrap

import pytest

from tests._scripts import run_python

_HELPERS = """
class Holder:
    pass


HELD = Holder()
HELD.last = None


def _remember(holder, n):
    holder.last = [n]


def _remember_twice_removed(holder, n):
    _remember(holder, n)


def _count(holder, n):
    holder.count = n


class Model:
    def __init__(self, depth):
        self.depth = depth

    def step(self, n):
        if self.depth == 1:
            _remember(HELD, n)
        elif self.depth == 2:
            _remember_twice_removed(HELD, n)
        else:
            _count(HELD, n)
        return n * 2
"""

_WALK = """
import json
import sys
import warnings

import cash
import helpers

c = cash.Cash(register_magic=False)
RUNS = []


@c.cache
def drive(model, n):
    RUNS.append(1)  # @cash:assume-safe
    return model.step(n)


class Runner:
    def go(self, model, n):
        RUNS.append(1)  # @cash:assume-safe
        return model.step(n)

    @classmethod
    @c.cache
    def go_on_class(cls, model, n):
        RUNS.append(1)  # @cash:assume-safe
        return model.step(n)


caller, depth = sys.argv[1], int(sys.argv[2])
call = {"function": drive, "bound method": c.cache(Runner().go), "classmethod": Runner.go_on_class}[caller]
warnings.simplefilter("ignore")
fresh = []
for n in (5, 5, 5):
    before = [n]
    helpers.HELD.last = before
    helpers.HELD.count = n
    assert call(helpers.Model(depth), n) == n * 2
    fresh.append(helpers.HELD.last == [n] and (depth == 3 or helpers.HELD.last is not before))
print(json.dumps({"runs": len(RUNS), "fresh": fresh}))
"""


@pytest.fixture
def walk(tmp_path):
    (tmp_path / "helpers.py").write_text(textwrap.dedent(_HELPERS), encoding="utf-8")
    (tmp_path / "walk.py").write_text(textwrap.dedent(_WALK), encoding="utf-8")

    def _run(caller: str, depth: int) -> dict:
        out = run_python("walk.py", caller, str(depth), cwd=tmp_path)
        return json.loads(out.stdout.strip().splitlines()[-1])

    return _run


@pytest.mark.parametrize("caller", ["function", "bound method", "classmethod"])
@pytest.mark.parametrize("depth", [1, 2])
def test_the_write_is_made_on_every_call_in_every_process(walk, caller, depth):
    assert walk(caller, depth) == {"runs": 3, "fresh": [True, True, True]}
    assert walk(caller, depth) == {"runs": 3, "fresh": [True, True, True]}


def test_a_write_that_leaves_the_same_object_still_hits(walk):
    """``holder.count = n`` with a small int leaves the object it found:
    plain Python and a hit leave the same state."""
    assert walk("function", 3) == {"runs": 1, "fresh": [True, True, True]}
