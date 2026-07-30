"""Real-kernel proof that a call unit's key is scoped to its ENCLOSING
STATEMENT, not just the call text (CAS-256).

**The bug.** ``call_cache_key``'s base key is built from the call's own
source and free names (``call_unit.py``), which says nothing about which
statement the call sits in. Two different statements whose call text and free
names happen to agree therefore share one base key -- and, inside a loop,
identical per-iteration ``loop_vars`` too -- so the second statement is served
the first's cached values::

    # cell 2
    for step in ['a', 'b', 'c']:
        vals[step] = fetch_next(conn)      # -> [('a', 1), ('b', 2), ('c', 3)]

    # cell 3
    for step in ['a', 'b', 'c']:
        other[step] = fetch_next(conn)     # served cell 2's values -- WRONG

Wrong on the FIRST run, no pre-existing cache required. The fix folds the
enclosing statement's identity (``ast.unparse`` of the statement, not its raw
source text -- see ``CallSite.stmt_identity``'s docstring for why the raw text
is a trap, CAS-242) into the call's key.

This file is the real-kernel arm. ``tests/test_notebook/test_call_unit_key.py``
and ``tests/test_notebook/test_call_interception_rewrite.py`` cover the same
property at the unit level (a unit test handing ``call_cache_key`` two
different ``stmt_identity`` values proves only that the parameter is read, not
that the production pipeline ever supplies two different ones for two real
statements) -- this file proves the production path actually produces the
discriminating value AND that neither of the two ways this fix could go wrong
(losing reorder-reuse, or losing per-iteration reuse within one statement) has
happened.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.loops]

# Above CallUnit's cost floor (10ms), or nothing is ever stored and every
# assertion below would hold whether or not this feature works.
_SLEEP = 0.15


def _defs(log):
    return (
        "import time, pathlib\n"
        f"LOG = pathlib.Path(r'{log}')\n"
        "state = {'n': 0}\n"
        "conn = object()\n"
        "def fetch_next(conn):\n"
        "    state['n'] += 1\n"
        "    with LOG.open('a') as fh:\n"
        "        fh.write('c\\n')\n"
        f"    time.sleep({_SLEEP})\n"
        "    return state['n']\n"
    )


def _n(log):
    return len(log.read_text().splitlines()) if log.exists() else 0


def test_two_statements_same_call_text_get_independent_call_caches(nb_runner, tmp_path):
    """The CAS-256 bug, reproduced live, with a cash-off oracle.

    With cash off (or before this fix), ``fetch_next`` runs 6 times total and
    the second loop reads its OWN 3 values, ``[('a', 4), ('b', 5), ('c', 6)]``
    -- state['n'] keeps advancing across both loops since neither hits a
    cache. The bug served the SECOND loop the FIRST loop's cached values
    instead, so it read ``[('a', 1), ('b', 2), ('c', 3)]`` with only 3 real
    calls total.

    Mutation that must make this fail: in ``call_interception.py``'s
    ``wrap_eligible_calls``, drop ``stmt_identity=stmt_identity,`` from the
    ``CallSite(...)`` construction (reverting to the pre-fix site shape).
    Verified by hand: with that line removed, ``SECOND`` reads
    ``[('a', 1), ('b', 2), ('c', 3)]`` and only 3 real calls are recorded.
    """
    log = tmp_path / "calls.log"
    first = (
        "vals = {}\n"
        "# @cash:cache-calls\n"
        "for step in ['a', 'b', 'c']:\n"
        "    vals[step] = fetch_next(conn)\n"
        "print('FIRST', sorted(vals.items()))\n"
    )
    second = (
        "other = {}\n"
        "# @cash:cache-calls\n"
        "for step in ['a', 'b', 'c']:\n"
        "    other[step] = fetch_next(conn)\n"
        "print('SECOND', sorted(other.items()))\n"
    )
    nb_runner.create_notebook([_defs(log), first, second])
    nb_runner.start_kernel()
    nb_runner.run_all()

    assert "FIRST [('a', 1), ('b', 2), ('c', 3)]" in nb_runner.get_output(2)
    assert "SECOND [('a', 4), ('b', 5), ('c', 6)]" in nb_runner.get_output(3), (
        nb_runner.get_output(3)
    )
    assert _n(log) == 6, (
        f"expected 6 real fetch_next() executions (3 per loop, no cross-loop "
        f"sharing), got {_n(log)}"
    )


def test_reorder_within_one_statement_still_reuses_cached_calls(nb_runner, tmp_path):
    """Negative #1: folding in the statement's identity must not reintroduce
    CAS-242 (order-dependence) for the statement it now discriminates.

    Reordering the SAME loop's items must still reuse every item's cached
    call -- the whole selling point of per-call, loop-var-keyed caching.
    ``loop_vars`` (unaffected by this fix) already makes each item's key
    order-independent; this pins that ``stmt_identity`` -- constant across
    iterations of one statement -- does not accidentally undo it.

    Mutation that must make this fail: in ``call_interception.py``, change
    ``stmt_identity = ast.unparse(stmt)`` to
    ``stmt_identity = ast.unparse(stmt) + str(id(stmt))``. Reordering re-runs
    the whole loop, since every fresh parse of the reordered cell produces a
    new AST object and therefore a new identity for every item, not just the
    changed ones. Verified by hand: with that mutation, the reorder below
    causes 3 new real calls instead of 0.
    """
    log = tmp_path / "calls.log"
    loop_code = (
        "vals = {{}}\n"
        "# @cash:cache-calls\n"
        "for step in {order}:\n"
        "    vals[step] = fetch_next(conn)\n"
        "print('OUT', sorted(vals.items()))\n"
    )
    nb_runner.create_notebook([_defs(log), loop_code.format(order=['a', 'b', 'c'])])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _n(log) == 3, "baseline did not run all three iterations"

    before = _n(log)
    nb_runner.set_cell_source(2, loop_code.format(order=['c', 'b', 'a']))
    nb_runner.run_cell(2)
    assert _n(log) - before == 0, (
        f"reordering re-ran cached calls ({_n(log) - before} new calls) -- "
        "stmt_identity must be constant across a reorder, exactly like it is "
        "across iterations"
    )


# Negative #2 ("two iterations of the SAME statement still share a statement
# identity") is deliberately NOT proven with a real-kernel rerun test here.
# Two shapes were tried by hand and both turned out to prove nothing: a
# byte-identical rerun, and a rerun with only a harmless space added to the
# loop header, are both restored by cash as ONE whole-statement cache hit --
# the loop is never re-decomposed into iterations and `stmt_identity` is never
# re-derived at all, so the `id(stmt)`-tagging mutation described on
# `test_reorder_within_one_statement_still_reuses_cached_calls` above leaves
# both scenarios unaffected (0 new calls whether or not the fix is broken). A
# third shape -- appending a new item to the iterable -- DOES force
# re-decomposition, but the three pre-existing items are then restored through
# a separate incremental-loop-extension path that does not appear to revisit
# `wrap_eligible_calls` for them either, so it was equally silent under the
# same mutation. The reorder test above is the one real-kernel shape that
# genuinely forces a fresh per-iteration re-derivation of `stmt_identity` for
# EVERY item and was confirmed (by hand) to catch the mutation -- it is the
# real-kernel evidence for this property. The direct, unit-level proof that
# two independent derivations of one statement's identity agree lives in
# `tests/test_notebook/test_call_interception_rewrite.py::
# test_same_statement_reparsed_twice_gets_the_same_stmt_identity`.
