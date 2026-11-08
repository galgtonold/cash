"""A call result its statement only binds is digested once.

``G = build_graph(edges)`` pickled the graph twice, back to back: the
call's store digested its result, and the statement then digested ``G``
again for its session hash. The two digests are the same function of the
same bytes (`stored_pickle.object_digest`), and nothing but binding the
name runs between them, so the statement takes the call's.

What must hold: the statement's digest is the one `compute_hash` gives the
value; a statement doing more than binding the result (``G = f(); G.x = 1``
in one expression is not possible, but ``g = touch(build())`` is) reads the
value again; a result served from the cache is digested by the statement as
before.
"""

from __future__ import annotations

import pytest

from cash.notebook import call_effects
from cash.notebook.statement import processor
from cash.value_hash import compute_hash
from tests._cell_driver import run_cash_cell

PICKLED: list[int] = []


class Graphish:
    """An object of a class of its own holding dicts, as a networkx graph
    does (no array: a result with arrays its statement only binds is not
    digested by its call, `CallEntries._too_big_to_digest`), counted each
    time it is pickled.

    Its nodes are named, not numbered: without the C extension a digest
    looks for sets by the bytes a set is pickled as (`stored_pickle._Sink`),
    and node 38033 (``0x9491``) is pickled as those bytes. Its digest is
    then taken over a second pickle with sets sorted, by design, in the
    call's own store, where this test counts what the statement adds."""

    def __init__(self) -> None:
        self.adj = {f"n{i}": {f"n{i + 1}": {"weight": 1.0}} for i in range(100_000)}
        self.tags = {"a": 1}

    def __reduce_ex__(self, protocol):
        PICKLED.append(1)
        return object.__reduce_ex__(self, protocol)


SETUP = f"""
import time
from {__name__} import Graphish

def build():
    time.sleep(0.3)
    return Graphish()

def touch(g):
    g.tags["b"] = 2
    return g
"""


@pytest.fixture
def magics(cash_magics, monkeypatch):
    # The statement takes the call's digest only while no other thread could
    # change the value in between (`trace_recorder.other_thread_alive`, asked
    # as the call stores and as the statement ends). A thread an earlier test
    # left running can start one while this test runs, which the suite's
    # host threads do not cover: then the statement rightly read the value
    # again. What this test pins is the case with no other thread.
    for module in (call_effects, processor):
        monkeypatch.setattr(module, "other_thread_alive", lambda _ns: False)
    run_cash_cell(cash_magics, SETUP)
    PICKLED.clear()
    return cash_magics


def _session_hash(magics, name):
    return magics.tracking_state.current_session_hashes.get(name)


def test_the_statement_takes_the_digest_its_call_took(magics, mock_shell):
    run_cash_cell(magics, "G = build()")
    assert PICKLED == [1], "the call's store and the statement each pickled the result"
    assert _session_hash(magics, "G") == compute_hash(mock_shell.user_ns["G"])


def test_a_statement_that_changes_the_result_reads_it_again(magics, mock_shell):
    run_cash_cell(magics, "g = touch(build())")
    assert _session_hash(magics, "g") == compute_hash(mock_shell.user_ns["g"])
    assert mock_shell.user_ns["g"].tags == {"a": 1, "b": 2}
