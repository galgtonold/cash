"""The `loop_vars` route from `for_handler.py` to `CallUnit._build_key` (CAS-243).

`call_cache_key`'s `loop_vars` parameter is fully covered at the unit level
(`test_call_unit_key.py`) -- given a dict, it discriminates correctly. What
was NOT covered until this file: whether the production pipeline ever
actually PRODUCES that dict and gets it to the call at invocation time. The
route is a stack-shaped attribute on `StatementProcessor`
(`_call_unit_loop_vars`), pushed/popped by `ForLoopHandler._process_one_iteration`
around each iteration's body statements and read by `CallUnit._build_key`
through the `loop_vars_provider` callable threaded via `CallCache`.

A test that constructs `CallUnit`/`CallCache` directly and hands `loop_vars`
straight to `call_cache_key` proves nothing about this wiring -- it never
touches `for_handler.py`, `StatementProcessor.loop_vars_scope`, or the
`loop_vars_provider` plumbing at all. Every test below goes through the REAL
production pipeline (`CashMagics.cash()` -> `StatementProcessor` ->
`ForLoopHandler` -> `CallCache`/`CallUnit`), the same one
`test_badge_sub_units.py`'s `MockShell` exercises for the loop-header
stamping wiring.
"""
from __future__ import annotations

import ast
from unittest.mock import MagicMock

import pytest
from traitlets.config import Configurable

from cash.backends import InMemoryBackend
from cash.core import Cash
from cash.notebook.cache_status import CacheStatus
from cash.notebook.ipython.magics import CashMagics
from cash.notebook.statement import StatementProcessor


class MockShell(Configurable):
    """Same shape as `test_badge_sub_units.py`'s `MockShell` -- runs the real
    production pipeline with only IPython's shell mocked out."""

    def __init__(self):
        super().__init__()
        self.user_ns = {}
        self.input_transformers_cleanup = []
        self.run_cell = MagicMock()
        self.events = MagicMock()
        self.ast_transformers = []
        self.user_global_ns = self.user_ns
        self.display_pub = type('MockDisplayPub', (), {'publish': MagicMock()})()


@pytest.fixture
def magics_fixture():
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MockShell()
    magics = CashMagics(shell, cash)
    magics._auto_cache_enabled = True
    yield magics, shell, backend
    backend.clear()
    shell.user_ns.clear()


# Split into a "defs" cell and a "loop" cell -- like a real notebook, and
# critically unlike re-running one cell that also re-executes
# `counter = {'n': 0}` every time (which would reset the counter regardless
# of whether `fetch_next` itself was cached, confounding the rerun test
# below). Only `_LOOP_CELL` is re-run to test cache reuse.
#
# Subscript-assignment body (`results[t] = fetch_next(conn)`), NOT
# `out.append(fetch_next(conn))` -- the append shape is an `ast.Expr(Call)`
# body and matches the accumulator single-unit fast path
# (`control_structures/processor.py`), which never reaches per-iteration
# decomposition (`test_badge_sub_units.py` documents this same trap). An
# `ast.Assign` body forces the real `ForLoopHandler` per-iteration route,
# which is what pushes/pops `loop_vars`.
#
# `conn` is a bare Name whose VALUE never changes across iterations -- its
# lineage is therefore one constant hash for all three, exactly the "no
# argument-side discriminator" shape `loop_vars` exists to cover. The hidden
# state lives in `counter`, a dict mutated from INSIDE `fetch_next`'s body --
# invisible to the call site's own free names (`fetch_next`, `conn`) and to
# `_hash_args`'s before/after check (which only watches the live arguments,
# and `conn` itself is never mutated).
_DEFS_CELL = """
import time
counter = {'n': 0}
conn = 'db-connection'
def fetch_next(conn):
    counter['n'] += 1
    time.sleep(0.02)
    return counter['n']

results = {}
"""

_LOOP_CELL = """
# @cash:cache-calls
for t in [1, 2, 3]:
    results[t] = fetch_next(conn)
"""


def test_hidden_state_call_gets_a_distinct_value_per_iteration(magics_fixture):
    """The bug this task exists to fix, reproduced through the real pipeline.

    Without `loop_vars` reaching the key, all three iterations build an
    IDENTICAL key (`conn`'s lineage never moves, and there is no computed
    argument expression to hash) -- so iteration 1 computes and stores, and
    iterations 2 and 3 both HIT iteration 1's entry. `results` would read
    `{1: 1, 2: 1, 3: 1}` instead of the correct `{1: 1, 2: 2, 3: 3}`, wrong on
    the very first run, no pre-existing cache required.

    Mutation that must make this fail: `call_unit.py`'s `_build_key` reverted
    to `loop_vars={}` (the TODO'd-out state this task replaced). Verified by
    hand: with that reversion this assertion fails with
    `{1: 1, 2: 1, 3: 1} != {1: 1, 2: 2, 3: 3}`.
    """
    magics_obj, shell, backend = magics_fixture
    magics_obj.cash("", _DEFS_CELL.strip())
    magics_obj.cash("", _LOOP_CELL.strip())
    assert shell.user_ns['results'] == {1: 1, 2: 2, 3: 3}
    assert shell.user_ns['counter']['n'] == 3, (
        "fetch_next() ran a different number of times than there were "
        "iterations -- either under-called (a false hit) or over-called "
        "(caching never engaged)"
    )


def test_hidden_state_call_reuses_cache_on_rerun(magics_fixture):
    """Re-running the identical loop cell must replay the SAME three values
    from cache, not recompute -- `counter['n']` must stop advancing.

    This is the reuse half of the same bug: a wiring mistake that pushed the
    wrong (e.g. stale, or unpopped) loop_vars could still pass the single-run
    test above by accident while breaking a rerun.
    """
    magics_obj, shell, backend = magics_fixture
    magics_obj.cash("", _DEFS_CELL.strip())
    magics_obj.cash("", _LOOP_CELL.strip())
    assert shell.user_ns['results'] == {1: 1, 2: 2, 3: 3}
    assert shell.user_ns['counter']['n'] == 3

    magics_obj.cash("", _LOOP_CELL.strip())
    assert shell.user_ns['results'] == {1: 1, 2: 2, 3: 3}, (
        "a rerun produced different values -- the per-iteration cache did "
        "not reuse cleanly"
    )
    assert shell.user_ns['counter']['n'] == 3, (
        "fetch_next() ran again on the rerun instead of hitting cache"
    )


# --------------------------------------------------------- stack safety

def _bare_statement_processor():
    backend = InMemoryBackend()
    cash = Cash(backend=backend, register_magic=False)
    shell = MagicMock()
    shell.user_ns = {}
    return StatementProcessor(cash_instance=cash, shell=shell, debug=False)


def test_loop_vars_scope_pops_even_when_the_body_raises():
    """`for_handler.py` relies on `finally` to restore the stack when a loop
    body statement raises (a body statement error propagates as an exception
    out of `_process_one_iteration`, and the NEXT thing to run in the same
    kernel -- another cell -- must not inherit a stale loop context).

    Mutation that must make this fail: dropping `finally` in
    `StatementProcessor.loop_vars_scope` (a bare pop after `yield`, unreached
    on an exception). Verified by hand: with that change this test's second
    assertion fails (`current_loop_vars()` still returns `{'t': 1}` instead
    of `{}` after the `with` block exits via the exception).
    """
    proc = _bare_statement_processor()
    assert proc.current_loop_vars() == {}

    with pytest.raises(ValueError):
        with proc.loop_vars_scope({'t': 1}):
            assert proc.current_loop_vars() == {'t': 1}
            raise ValueError("body statement blew up")

    assert proc.current_loop_vars() == {}, (
        "loop_vars_scope leaked a stack entry across an exception"
    )


def test_loop_vars_scope_nests_innermost_wins():
    """A nested loop's push must shadow the outer one, and popping must
    restore the outer value exactly -- not clear it.

    Mirrors what `build_iteration_context` already does at the VALUE level
    (merging `parent_context` forward): the inner loop's own `loop_vars`
    dict is expected to already carry the outer vars merged in, so this test
    is about the STACK mechanics (LIFO push/pop), not re-testing the merge.
    """
    proc = _bare_statement_processor()
    with proc.loop_vars_scope({'outer': 1}):
        assert proc.current_loop_vars() == {'outer': 1}
        with proc.loop_vars_scope({'outer': 1, 'inner': 2}):
            assert proc.current_loop_vars() == {'outer': 1, 'inner': 2}
        assert proc.current_loop_vars() == {'outer': 1}, (
            "popping the inner loop's vars did not restore the outer loop's"
        )
    assert proc.current_loop_vars() == {}


# --------------------------------------------------------- for_handler.py's own guard

class _ShellStub:
    def __init__(self):
        self.user_ns = {}


class _StatementProcessorWithoutLoopVarsScope:
    """Deliberately lacks `loop_vars_scope` -- the shape `StatementProcessor`
    had before this task, or any future variant that hasn't picked up the
    method. `ForLoopHandler` must not assume it exists.
    """

    def __init__(self):
        self.variable_lineage: dict = {}
        self.vars_with_mutation_lineage: set = set()
        self.compute_hash = lambda v: 'fakehash'

    def process_statement(self, code, ttl, silent, annotation=None):
        return {
            'status': CacheStatus.COMPUTED,
            'execution_time': 0.01,
            'stdout': '',
            'stderr': '',
            'outputs': [],
        }


def test_for_loop_runs_even_when_statement_processor_lacks_loop_vars_scope():
    """Minor finding: `for_handler.py`'s `with
    self.statement_processor.loop_vars_scope(loop_vars):` must not assume the
    method exists. Unreachable in production today (`StatementProcessor` is
    the one and only class ever passed in, at `magics.py`'s single
    construction site) -- but `for_handler.py` is the file where a caching
    optimisation failing breaks the USER'S LOOP outright (an unhandled
    `AttributeError` propagates out of `handler.process()` as
    `success=False`, and `cell_executor.py` re-raises it as the user's own
    error), not merely its caching. Guarded with
    `getattr(..., None)` falling back to `contextlib.nullcontext()`.

    Mutation that must make this fail: revert the guard in `for_handler.py`
    to the bare `with self.statement_processor.loop_vars_scope(loop_vars):`.
    Verified by hand: with that reversion this test raises
    `AttributeError: '_StatementProcessorWithoutLoopVarsScope' object has no
    attribute 'loop_vars_scope'` instead of completing.
    """
    from cash.notebook.control_structures.for_handler import ForLoopHandler

    shell = _ShellStub()
    handler = ForLoopHandler(
        shell, _StatementProcessorWithoutLoopVarsScope(), debug=False, dispatcher=MagicMock(),
    )
    node = ast.parse("for x in [1, 2]:\n    y = x\n").body[0]

    result = handler.process(node, ttl=None, silent=True, parent_context=None)

    assert result.success is True, result.error
