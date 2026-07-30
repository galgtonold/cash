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


# --------------------------------------------------------- loop_var_digests stack
#
# The round-3 review fix (reading a loop var's digest from `variable_lineage`)
# was correct on the discrimination question but wrong on scope: that dict is
# flat, keyed only by name, and never popped, so a nested loop reusing an
# outer loop's target name left a STALE entry for the rest of the outer
# iteration. The replacement threads the digest through THIS stack instead --
# pushed/popped in lockstep with the value stack above, by the same
# `loop_vars_scope` call -- which has the scope discipline `variable_lineage`
# lacks. These tests cover the stack mechanics directly; the end-to-end
# correctness proof (a real nested loop, a real hidden-state call, a real
# kernel) lives in `test_call_unit_loop_vars_real_kernel.py`'s
# `test_nested_loop_reusing_the_target_name_*` / `test_sibling_loops_*` tests.

def test_loop_var_digests_scope_pops_in_lockstep_with_loop_vars():
    """The digests stack must be exception-safe too, popped by the SAME
    `finally` that pops the values stack -- not a second, independent
    mechanism that could desync from it.

    Mutation that must make this fail: in `loop_vars_scope`'s `finally`,
    drop the `self._call_unit_loop_var_digests.pop()` line (keep the values
    pop). Verified by hand: with that change this test's final assertion
    fails (`current_loop_var_digests()` still returns `{'t': 'digest-A'}`
    instead of `{}` after the exception propagates out of the `with` block).
    """
    proc = _bare_statement_processor()
    assert proc.current_loop_var_digests() == {}

    with pytest.raises(ValueError):
        with proc.loop_vars_scope({'t': 1}, {'t': 'digest-A'}):
            assert proc.current_loop_var_digests() == {'t': 'digest-A'}
            raise ValueError("body statement blew up")

    assert proc.current_loop_var_digests() == {}, (
        "loop_var_digests_scope leaked a stack entry across an exception"
    )


def test_loop_var_digests_reused_name_resolves_to_the_current_scope():
    """The exact stack-discipline property the real-kernel repro exercises,
    reproduced directly against the stack (no notebook, no kernel boot): an
    inner scope reusing the SAME name as an outer scope must not leak its
    digest into the outer scope once popped.

    Mutation that must make this fail: `current_loop_var_digests` reverted
    to `self._call_unit_loop_var_digests[-1] if ... else {}` (top-of-stack
    only, mirroring `current_loop_vars`'s VALUE-stack implementation) instead
    of merging across the whole stack -- this happens to still pass THIS
    specific test (the inner scope is already fully popped by the time of
    the final assertion, so top-of-stack and merge agree here), which is
    exactly why the assertion INSIDE the `with proc.loop_vars_scope(...)`
    block below is load-bearing: swap `'t': 'inner-digest'` for a check that
    an OUTER name (not reused by the inner scope) is still resolvable while
    the inner scope is active, and a top-of-stack-only implementation fails
    it.
    """
    proc = _bare_statement_processor()
    with proc.loop_vars_scope({'t': 1, 'outer_only': 99}, {'t': 'outer-digest', 'outer_only': 'outer-only-digest'}):
        assert proc.current_loop_var_digests() == {
            't': 'outer-digest', 'outer_only': 'outer-only-digest',
        }
        with proc.loop_vars_scope({'t': 2}, {'t': 'inner-digest'}):
            # Inner scope's OWN reused name shadows the outer's.
            assert proc.current_loop_var_digests()['t'] == 'inner-digest'
            # An OUTER-only name (not reused/re-pushed by the inner scope)
            # must still resolve -- proves this is a MERGE across the whole
            # stack, not a top-of-stack-only lookup that would lose it.
            assert proc.current_loop_var_digests()['outer_only'] == 'outer-only-digest'
        # Inner scope popped -- 't' must resolve back to the OUTER's digest,
        # not linger at the inner's.
        assert proc.current_loop_var_digests()['t'] == 'outer-digest', (
            "the inner scope's digest for a reused name leaked past its pop"
        )
    assert proc.current_loop_vars() == {}


# --------------------------------------------------------- depth-keyed call-key scope (CAS-257 defect 1)
#
# `current_loop_vars()`/`current_loop_var_digests()` above are pinned to
# their EXACT pre-existing contract -- top-of-stack for values, bare-name
# merge-with-innermost-winning for digests -- and stay that way: other
# callers (and the tests above) read them directly and must not see any
# behaviour change. The CAS-257 fix lives in a SEPARATE read path,
# `_depth_keyed_loop_scope` (exposed via `current_loop_vars_for_call_key` /
# `current_loop_var_digests_for_call_key`), which `StatementProcessor` now
# wires into `CallCache` INSTEAD of the two methods above. These tests cover
# that path directly, at the stack level -- no notebook, no kernel -- mirroring
# the section above; the end-to-end proof (a real nested loop, a call INSIDE
# the reuse, a real kernel, a cash-off oracle) lives in
# `test_call_unit_loop_vars_real_kernel.py`'s
# `test_call_inside_a_name_reusing_inner_loop_*` tests.

def test_depth_keyed_scope_gives_a_reused_name_two_distinct_slots():
    """The exact shape CAS-257 defect 1 reports: ``for q in A: for q in B:
    <call>`` -- while BOTH scopes are simultaneously active (the call sits
    INSIDE the inner loop, not after it), the outer 'q' and the inner 'q'
    must occupy two different (depth, name) slots, not collide onto one
    bare 'q' the way `current_loop_vars()`/`current_loop_var_digests()`
    already do (see the sections above -- that collision is exactly what
    they are pinned to keep doing, for everything BUT this path).

    Mutation that must make this fail: revert
    `current_loop_vars_for_call_key`/`current_loop_var_digests_for_call_key`
    to delegate straight to `current_loop_vars`/`current_loop_var_digests`
    (undoing the CAS-257 fix). Verified by hand: with that reversion both
    calls return `{'q': 7}` / `{'q': 'digest-inner'}` -- only the inner
    scope survives -- and the assertions below fail (`'0:q'` is missing
    entirely from either dict).
    """
    proc = _bare_statement_processor()
    with proc.loop_vars_scope({'q': 'p'}, {'q': 'digest-outer'}):
        with proc.loop_vars_scope({'q': 7}, {'q': 'digest-inner'}):
            values = proc.current_loop_vars_for_call_key()
            digests = proc.current_loop_var_digests_for_call_key()
            assert values == {'0:q': 'p', '1:q': 7}
            assert digests == {'0:q': 'digest-outer', '1:q': 'digest-inner'}


def test_depth_keyed_scope_pop_restores_the_single_outer_slot():
    """Once the inner scope pops, only the outer's depth-0 slot remains --
    the mirror image of the collision test above, proving the fix does not
    leak an inner entry past its own pop (the same discipline
    `test_loop_var_digests_reused_name_resolves_to_the_current_scope`
    already requires of the un-keyed digest stack).
    """
    proc = _bare_statement_processor()
    with proc.loop_vars_scope({'q': 'p'}, {'q': 'digest-outer'}):
        with proc.loop_vars_scope({'q': 7}, {'q': 'digest-inner'}):
            pass
        assert proc.current_loop_vars_for_call_key() == {'0:q': 'p'}
        assert proc.current_loop_var_digests_for_call_key() == {'0:q': 'digest-outer'}
    assert proc.current_loop_vars_for_call_key() == {}
    assert proc.current_loop_var_digests_for_call_key() == {}


def test_depth_prefix_is_positional_not_order_dependent():
    """Depth is WHERE a call sits in the loop nesting (the stack's length at
    the moment of the push), not WHICH iteration or in what sequence the
    iterable was walked. So the SAME call site contributes an entry under
    the SAME depth-keyed name regardless of which value is bound there --
    only the value/digest attached to that name varies. This is the
    property `test_loop_reorder_reuse_claim.py` / the CAS-257 write-up rely
    on to rule out reintroducing CAS-242 (reordering an iterable must not
    change a call's key beyond the value it actually reads): a depth-keyed
    name is never a stand-in for iteration order or an execution counter.

    Two INDEPENDENT top-level scopes below (the second opens only after the
    first has fully popped) simulate two different outer-loop iterations,
    as if the iterable had been walked in a different order, or the loop
    had simply advanced -- either way, the call site's own nesting depth is
    unchanged, so both must resolve to the identical name '0:q'.

    Mutation that must make this fail: key by a running push COUNTER instead
    of stack depth (never decremented on pop) -- the second scope below would
    then land on '1:q' instead of '0:q' merely because a scope ran before it,
    which is exactly the per-run execution-counter mistake `call_cache_key`'s
    own docstring already rejects for `arg_digests` (the removed
    `repeat_index` design). Verified by hand: with such a counter this test's
    `set(...) == {'0:q'}` assertions fail for the second scope (`{'1:q'}`
    instead), even though it is the SAME call site.
    """
    proc = _bare_statement_processor()
    with proc.loop_vars_scope({'q': 'p'}, {'q': 'digest-p'}):
        first = proc.current_loop_var_digests_for_call_key()
    with proc.loop_vars_scope({'q': 'r'}, {'q': 'digest-r'}):
        second = proc.current_loop_var_digests_for_call_key()

    assert set(first) == {'0:q'}
    assert set(second) == {'0:q'}
    assert first['0:q'] != second['0:q'], (
        "different values at the same depth must still discriminate"
    )


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
