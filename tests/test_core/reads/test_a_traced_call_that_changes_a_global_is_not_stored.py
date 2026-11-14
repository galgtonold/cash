"""A traced call that changes a global its code reads is not stored.

A call that picks code by a run-time name is checked on a trace of what it
ran (`cash.decorator.code_trace`), whose items hold the globals that code
reads, digested once the call ended. A call that changed one of them in
place through a library or C function (``heapq.heappush(HEAP, y)``,
``D.rotate(1)``, ``np.copyto(ARR, y)``) left no finding in the code's
source, and its entry was served: a second call in the process skipped the
change, so ``HEAP`` held one item where running the program without cash
leaves two. Those globals are digested as the code reading them starts too,
and a call that moved one is not stored.

Throwing a return value away is not itself a reason: a check that only
raises (``check_targets(y)``) is stored and served in a new process, and a
callee that appends to a module-level list is still refused, on that
finding. A draw from the global ``random`` generator is keyed on its state
and replayed on a hit, so the program goes on as without cash.
"""

from __future__ import annotations

import json
import sys
import textwrap

import pytest

from tests._scripts import run_python

pytestmark = pytest.mark.skipif(sys.version_info < (3, 12), reason="sys.monitoring is Python 3.12+")

np = pytest.importorskip("numpy")

_CHECKS = """
import collections
import heapq
import operator
import random

import numpy as np


def check_targets(y):
    if not isinstance(y, int):
        raise TypeError("targets must be an int")
    _check_sign(y)


def _check_sign(y):
    if y < 0:
        raise ValueError("targets must not be negative")


HEAP = []
D = collections.deque([1, 2, 3])
L = [1, 2, 3, 4, 5]
ARR = np.zeros(3)
G = {}
SEEN = []


def remember(y):
    SEEN.append(y)


def pure(y):
    check_targets(y)
    return y * 2


def heappush(y):
    heapq.heappush(HEAP, y)
    return y * 2


def rotate(y):
    D.rotate(1)
    return y * 2


def shuffle(y):
    random.shuffle(L)
    return y * 2


def copyto(y):
    np.copyto(ARR, y)
    return y * 2


def fill(y):
    ARR.fill(y)
    return y * 2


def setitem(y):
    operator.setitem(G, "k", G.get("k", 0) + y)
    return y * 2


def draw(y):
    x = [1, 2, 3, 4, 5]
    random.shuffle(x)
    return x


def remembered(y):
    remember(y)
    return y * 2


def state():
    return {
        "HEAP": HEAP,
        "D": list(D),
        "L": L,
        "ARR": ARR.tolist(),
        "G": G,
        "SEEN": SEEN,
        "next": random.random(),
    }
"""

_MAIN = """
import json
import random
import sys

import cash
import checks

method, times, plain = sys.argv[1], int(sys.argv[2]), sys.argv[3] == "plain"


def run(method, y):
    print("ran")
    return getattr(checks, method)(y)


if not plain:
    run = cash.cache(run)

random.seed(0)
results = [run(method, 4) for _ in range(times)]
reasons = [] if plain else [w["message"] for w in run.cache_info()["warnings"] if w["code"] == "KEY-UNTRACKABLE-DEP"]
print(json.dumps({"results": results, "state": checks.state(), "reasons": reasons}))
"""


@pytest.fixture
def project(tmp_path):
    (tmp_path / "checks.py").write_text(textwrap.dedent(_CHECKS), encoding="utf-8")
    (tmp_path / "main.py").write_text(textwrap.dedent(_MAIN), encoding="utf-8")
    return tmp_path


def _run(project, method: str, times: int = 1, plain: bool = False) -> tuple[int, dict]:
    lines = run_python("main.py", method, str(times), "plain" if plain else "cached", cwd=project).stdout.splitlines()
    return lines.count("ran"), json.loads(lines[-1])


_EFFECTS = ["heappush", "rotate", "shuffle", "copyto", "fill", "setitem"]


@pytest.mark.parametrize("method", _EFFECTS)
def test_calling_twice_in_one_process_changes_the_global_twice(project, method):
    ran, got = _run(project, method, times=2)
    _, plain = _run(project, method, times=2, plain=True)
    assert got["state"] == plain["state"]
    assert got["results"] == plain["results"]
    assert ran == 2, "the second call was served and skipped the change"
    assert any("modifies the module global 'checks." in r for r in got["reasons"]), got["reasons"]


@pytest.mark.parametrize("method", _EFFECTS)
def test_a_new_process_changes_the_global_again(project, method):
    _, plain = _run(project, method, plain=True)
    for _ in range(2):
        ran, got = _run(project, method)
        assert ran == 1
        assert got["state"] == plain["state"]


def test_a_heap_holds_both_pushes(project):
    _, got = _run(project, "heappush", times=2)
    assert got["state"]["HEAP"] == [4, 4]


def test_a_dropped_check_is_stored_and_served_in_a_new_process(project):
    ran, got = _run(project, "pure")
    assert (ran, got["results"], got["reasons"]) == (1, [8], [])
    ran, got = _run(project, "pure")
    assert (ran, got["results"]) == (0, [8])


def test_a_dropped_call_to_a_function_appending_to_a_global_is_refused(project):
    _, plain = _run(project, "remembered", times=2, plain=True)
    ran, got = _run(project, "remembered", times=2)
    assert ran == 2
    assert got["state"]["SEEN"] == plain["state"]["SEEN"] == [4, 4]
    assert any("SEEN.append()" in r for r in got["reasons"]), got["reasons"]


def test_a_draw_from_the_global_generator_leaves_it_where_the_program_would(project):
    _, plain = _run(project, "draw", times=2, plain=True)
    for _ in range(2):
        _, got = _run(project, "draw", times=2)
        assert got["results"] == plain["results"]
        assert got["state"]["next"] == plain["state"]["next"]


_WORKER = """
import heapq

HEAP = []


class Worker:
    def push(self, y):
        heapq.heappush(HEAP, y)
        return y * 2
"""

_BY_METHOD = """
import json
import sys

import cash
import worker


@cash.cache
def run(obj, name, y):
    print("ran")
    return getattr(obj, name)(y)


w = worker.Worker()
results = [run(w, sys.argv[1], 4) for _ in range(2)]
reasons = [m["message"] for m in run.cache_info()["warnings"] if m["code"] == "KEY-UNTRACKABLE-DEP"]
print(json.dumps({"results": results, "HEAP": worker.HEAP, "reasons": reasons}))
"""


def test_a_method_of_an_arguments_class_changing_a_global_is_checked_too(tmp_path):
    """The code of a method an argument's class holds is in the key
    (`code_trace.folded_codes`) and its findings are not judged again; the
    globals it reads are still checked for a change the call made."""
    (tmp_path / "worker.py").write_text(textwrap.dedent(_WORKER), encoding="utf-8")
    (tmp_path / "by_method.py").write_text(textwrap.dedent(_BY_METHOD), encoding="utf-8")
    for _ in range(2):
        lines = run_python("by_method.py", "push", cwd=tmp_path).stdout.splitlines()
        got = json.loads(lines[-1])
        assert lines.count("ran") == 2
        assert got["HEAP"] == [4, 4]
        assert any("modifies the module global 'worker.HEAP'" in r for r in got["reasons"]), got["reasons"]
