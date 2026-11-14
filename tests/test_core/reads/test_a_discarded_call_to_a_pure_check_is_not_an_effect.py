"""A traced call that drops what a pure check returns is still stored.

A call that picks code by a run-time name is judged on the code it ran
(`cash.decorator.code_trace`), and a call whose return is thrown away is
read as one made for its effect. A check (``check_targets(y)``) returns
nothing and raises on a bad argument: when the purity analysis finds no
effect in the function a discarded call names, a hit skips nothing, and
the call is stored. The check stays in the trace, so an edit to it runs
the call again; a discarded call to a function that writes a global or
changes its argument still keeps the call from being stored.
"""

from __future__ import annotations

import sys
import textwrap

import pytest

from tests._scripts import run_python

pytestmark = pytest.mark.skipif(sys.version_info < (3, 12), reason="sys.monitoring is Python 3.12+")

_CHECKS = """
def check_targets(y):
    if not isinstance(y, int):
        raise TypeError("targets must be an int")
    _check_sign(y)


def _check_sign(y):
    if y < 0:
        raise ValueError("targets must not be negative")


COUNT = 0


def count(y):
    global COUNT
    COUNT += 1


def remember(seen, y):
    seen.append(y)


SEEN = []


def resample(y):
    check_targets(y)
    return y * 2


def resample_counted(y):
    count(y)
    return y * 2


def resample_remembered(y):
    remember(SEEN, y)
    return y * 2
"""

_MAIN = """
import sys
import cash
import checks


@cash.cache
def run(method, y):
    print("ran")
    return getattr(checks, method)(y)


print(run(sys.argv[1], 4))
print(" ".join(w["code"] for w in run.cache_info()["warnings"] if w["code"] == "KEY-UNTRACKABLE-DEP"))
"""


def _write(tmp_path, checks: str = _CHECKS) -> None:
    (tmp_path / "checks.py").write_text(textwrap.dedent(checks), encoding="utf-8")
    (tmp_path / "main.py").write_text(textwrap.dedent(_MAIN), encoding="utf-8")


def _run(tmp_path, method: str) -> list[str]:
    return run_python("main.py", method, cwd=tmp_path).stdout.split()


def test_a_function_dropping_a_pure_check_is_served_in_a_new_process(tmp_path):
    _write(tmp_path)
    assert _run(tmp_path, "resample") == ["ran", "8"]
    assert _run(tmp_path, "resample") == ["8"]


def test_an_edit_to_the_check_runs_the_call_again(tmp_path):
    _write(tmp_path)
    assert _run(tmp_path, "resample") == ["ran", "8"]
    _write(tmp_path, _CHECKS.replace("if y < 0:", "if y < -1:"))
    assert _run(tmp_path, "resample") == ["ran", "8"]
    assert _run(tmp_path, "resample") == ["8"]


@pytest.mark.parametrize("method", ["resample_counted", "resample_remembered"])
def test_a_dropped_call_with_an_effect_is_still_not_stored(tmp_path, method):
    _write(tmp_path)
    assert _run(tmp_path, method) == ["ran", "8", "KEY-UNTRACKABLE-DEP"]
    assert _run(tmp_path, method) == ["ran", "8", "KEY-UNTRACKABLE-DEP"]
