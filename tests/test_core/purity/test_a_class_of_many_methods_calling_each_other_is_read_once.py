"""An object whose class has many methods calling one another is read once
per method, not once per path through them.

Whether a method changes its object (``self.n += 1``), itself or through a
method of the class it calls on it, was answered by walking every path from
it, parsing each method's source again on each one. A cached function handed
an object of a strategy class whose 30 methods call one another paid 900
parses per call: 130 ms on every hit. Each method's source is now read once,
and each method reached is looked at once per question.

What is found stays the same: a change thirty calls down the chain still
counts, the class is read as it is now (a method rebound on it after the
first question is the one the next question reads), and a chain that changes
nothing is still found inert.
"""

from __future__ import annotations

import linecache

import pytest

import cash.analysis.handed_callables as hc
from cash.analysis.callee_effects import parse_function_source
from cash.analysis.handed_callables import hands_a_state_changing_callable

pytestmark = pytest.mark.core

N = 30


def _source(last: str) -> str:
    methods = "".join(
        f"    def helper_{i}(self, x):\n"
        f"        if x > {i}:\n"
        f"            return self.helper_{i + 1}(x - 1) + self.helper_{(i + 2) % N}(x - 2)\n"
        f"        return x * self.RATE\n"
        for i in range(N - 1)
    )
    return (
        "LOG = []\n"
        "class Strategy:\n"
        "    RATE = 2\n"
        f"{methods}"
        f"    def helper_{N - 1}(self, x):\n"
        f"        {last}\n"
        "        return x\n"
        "def run(s, k):\n"
        "    return s.helper_0(k)\n"
    )


def _namespace(code: str) -> dict:
    filename = f"<many-methods-{abs(hash(code))}>"
    linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
    namespace: dict = {"__name__": "__main__"}
    exec(compile(code, filename, "exec"), namespace)
    return namespace


def _changes(ns: dict) -> bool:
    cls = ns["Strategy"]
    fn = cls.helper_0
    return hc._changes_its_first_argument(fn, cls, hc._source_of(fn))


def test_a_change_at_the_end_of_the_chain_is_found():
    assert _changes(_namespace(_source("self.n = x")))


def test_a_chain_that_changes_nothing_is_inert():
    assert not _changes(_namespace(_source("pass")))


def test_a_method_rebound_on_the_class_is_read_as_it_is_now():
    ns = _namespace(_source("pass"))
    assert not _changes(ns)
    edited = _namespace("def helper(self, x):\n    self.seen = x\n    return x\n")
    ns["Strategy"].helper_29 = edited["helper"]
    assert _changes(ns)


def test_a_global_changed_at_the_end_of_the_chain_still_runs_the_call():
    ns = _namespace(_source("LOG.append(x)"))
    assert hands_a_state_changing_callable(ns["run"], (ns["Strategy"](), 3), {})
    pure = _namespace(_source("pass"))
    assert not hands_a_state_changing_callable(pure["run"], (pure["Strategy"](), 3), {})


def test_each_method_is_parsed_once_however_many_paths_reach_it(monkeypatch):
    ns = _namespace(_source("pass"))
    parsed = [0]

    def counting(source):
        parsed[0] += 1
        return parse_function_source(source)

    monkeypatch.setattr(hc, "parse_function_source", counting)
    for _ in range(5):
        assert not hands_a_state_changing_callable(ns["run"], (ns["Strategy"](), 3), {})
    # Each of the N methods (and run) is parsed at most a few times in all,
    # not once per path from each method (hundreds a question before).
    assert parsed[0] <= 3 * (N + 1)
