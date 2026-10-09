"""The body of a notebook function a statement calls is read from its source
once per definition, not on every statement that calls it.

Each statement is walked for the module state and for the process state it
sets, both following the notebook functions it calls into their bodies; each
read the function's source again (``inspect.getsource``: a tokenize per
read). Pins the work, not the behaviour: a refactor may expect this to fail.
"""

from __future__ import annotations

import inspect
import os

from cash.notebook import callee_reach


def set_mode():
    os.environ["CASH_TEST_PC_MODE"] = "b"


def plain_helper(x):
    return x + 1


def _source_reads(monkeypatch) -> list[object]:
    callee_reach._code_body.cache_clear()
    callee_reach._code_source.cache_clear()
    reads: list[object] = []
    real = inspect.getsource
    monkeypatch.setattr(inspect, "getsource", lambda obj: reads.append(obj) or real(obj))
    return reads


def test_the_body_is_read_once_for_every_statement_calling_it(monkeypatch):
    reads = _source_reads(monkeypatch)
    namespace = globals()
    for _ in range(3):
        assert callee_reach.process_state_writes("set_mode()", namespace) == {callee_reach.ENVIRON}
        callee_reach.module_state_writes("y = plain_helper(1)", namespace)
    assert len(reads) <= 2  # once for each of the two functions


def test_a_new_definition_is_read_again(monkeypatch):
    reads = _source_reads(monkeypatch)
    namespace = dict(globals())
    callee_reach.process_state_writes("set_mode()", namespace)
    before = len(reads)
    exec(compile(inspect.getsource(set_mode), __file__, "exec"), namespace)  # a def run again
    reads.clear()
    callee_reach.process_state_writes("set_mode()", namespace)
    assert before >= 1
    assert len(reads) == 1
