"""A module-level ``functools.partial`` of the user's own function that the
cached function hands on instead of calling is keyed by what it binds.

``INT = partial(scale, rows=1)`` and a cached function that does
``list(map(INT, [x]))``: the name is read, never called, so no call site
queued it for the helper walk, and the partial's function and arguments
reached no key. Rebinding ``INT = partial(scale, rows=2)`` -- in the running
process, in a notebook cell, or by editing the line before the next run --
served the result computed with ``rows=1``. Calling ``INT(x)`` directly was
keyed all along; handing it on now is too.

What cannot be hashed is not dropped silently: the call stays cached and
says so once (KEY-UNHASHABLE-GLOBAL).
"""

from __future__ import annotations

import functools
import sys
import textwrap
import threading
import warnings

import pytest

from tests._cell_driver import run_cash_cell
from tests._scripts import run_python

pytestmark = pytest.mark.core

THIS = sys.modules[__name__]


def scale(x, rows):
    return x * rows


def scale_with(x, rows, lock=None):
    return x * rows


def apply(fn, x):
    return fn(x)


INT = functools.partial(scale, rows=1)
NESTED = functools.partial(functools.partial(scale), rows=1)
LOCKED = functools.partial(scale_with, rows=1, lock=threading.Lock())


def _warm(fn, x, times=4):
    """The same call repeated, so every memo that can serve it does."""
    return [fn(x) for _ in range(times)][-1]


@pytest.fixture
def handed(disk_cash):
    @disk_cash.cache
    def by_map(x):
        return list(map(INT, [x]))

    @disk_cash.cache
    def to_own_helper(x):
        return apply(INT, x)

    @disk_cash.cache
    def by_attribute(x):
        return list(map(THIS.INT, [x]))

    @disk_cash.cache
    def nested(x):
        return list(map(NESTED, [x]))

    return {"by_map": by_map, "to_own_helper": to_own_helper, "by_attribute": by_attribute, "nested": nested}


@pytest.mark.parametrize("shape", ["by_map", "to_own_helper", "by_attribute"])
def test_rebinding_the_partial_recomputes(handed, monkeypatch, shape):
    fn = handed[shape]
    assert _warm(fn, 3) in ([3], 3)
    monkeypatch.setattr(THIS, "INT", functools.partial(scale, rows=2))
    assert _warm(fn, 3) in ([6], 6)
    monkeypatch.setattr(THIS, "INT", functools.partial(scale, rows=1))
    assert _warm(fn, 3) in ([3], 3)


def test_a_partial_of_a_partial_rebound_recomputes(handed, monkeypatch):
    assert _warm(handed["nested"], 3) == [3]
    monkeypatch.setattr(THIS, "NESTED", functools.partial(functools.partial(scale), rows=5))
    assert _warm(handed["nested"], 3) == [15]


def test_an_argument_that_cannot_be_hashed_stays_cached_and_says_so(disk_cash):
    runs = []

    @disk_cash.cache
    def locked(x):
        runs.append(x)
        return list(map(LOCKED, [x]))

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert locked(3) == [3]
        assert locked(3) == [3]
    assert runs == [3]
    messages = [str(w.message) for w in caught]
    assert any("KEY-UNHASHABLE-GLOBAL" in m and "LOCKED" in m for m in messages), messages


MODULE = """
import functools
import cash

def scale(x, rows):
    return x * rows

INT = functools.partial(scale, rows={rows})

@cash.cache
def report(x):
    return list(map(INT, [x]))

print(report(3))
"""


def test_the_line_edited_before_the_next_run_recomputes(tmp_path):
    (tmp_path / "job.py").write_text(textwrap.dedent(MODULE.format(rows=1)), encoding="utf-8")
    assert run_python("job.py", cwd=tmp_path).stdout.split() == ["[3]"]
    (tmp_path / "job.py").write_text(textwrap.dedent(MODULE.format(rows=2)), encoding="utf-8")
    assert run_python("job.py", cwd=tmp_path).stdout.split() == ["[6]"]


def test_a_notebook_cell_rebinding_the_partial_recomputes(cash_magics):
    ns = cash_magics.shell.user_ns
    run_cash_cell(
        cash_magics,
        "import functools\nimport cash\ndef scale(x, rows):\n    return x * rows\n"
        "INT = functools.partial(scale, rows=1)",
    )
    run_cash_cell(cash_magics, "@cash.cache\ndef report(x):\n    return list(map(INT, [x]))")
    run_cash_cell(cash_magics, "a = report(3)\nb = list(map(INT, [3]))")
    assert (ns["a"], ns["b"]) == ([3], [3])
    run_cash_cell(cash_magics, "INT = functools.partial(scale, rows=2)")
    run_cash_cell(cash_magics, "a = report(3)\nb = list(map(INT, [3]))")
    assert (ns["a"], ns["b"]) == ([6], [6])
