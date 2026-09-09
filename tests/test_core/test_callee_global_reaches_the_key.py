"""A global read by a cached CALLEE must reach its caller's key.

Round-15 gate finding, reported against a library shaped the ordinary way --
`config.py` holds a constant, `io.py` reads a file, `build.py` calls both. The
tester changed the constant and `build` returned the answer computed under the
old value: **zero executions, no warning, and one process disagreeing with
itself**, since calling the callee directly in the same run gave the new answer.

A global can reach a cache key through three channels. Two of them worked:

* the cached function's own globals (``_fold_read_globals``),
* the globals its plain helpers read (``_fold_helper_read_globals``).

The third -- the globals a cached CALLEE reads -- was folded into that callee's
own key and nowhere else, so the caller never saw it move.

Two things measurement corrected in the ticket, both worth keeping:

* **The file read was a red herring.** The tester's control table isolated "the
  callee reads a file" as the trigger, but the file only made the work
  expensive enough to be persisted past RAM. The no-file arm looked correct
  because its result was too cheap to store, so every run recomputed anyway.
  Against a backend that stores everything, the no-file shape is stale too.
* **The SOURCE side of the same edge always worked.** Editing the callee's body,
  or the cross-module helper's body, invalidates the caller through the graph
  and the transitive helper hashes. Only the DATA those functions read was
  invisible -- which is also why ``depends_on=``, a source-level declaration,
  could not rescue it.

The functions live in ``_callee_calc``, a real module, because the call edge is
recorded by reading a module's source: cached functions defined inside a test
function have no recorded dependencies at all, so the fold under test is never
reached and the fixture passes against the unfixed code. The constant is
mutated in-process, which is exact -- the fold re-reads the module live from
``sys.modules`` on every call, so an attribute assignment is what a library
upgrade looks like from cash's side.
"""
from __future__ import annotations

import pytest

from . import _callee_calc, _callee_rules


@pytest.fixture(autouse=True)
def clean_library():
    """A cold cache, an empty run log, and the constants put back."""
    before = (_callee_rules.THRESHOLD, _callee_rules.UNUSED_SETTING)
    _callee_calc.c.backend.clear()
    _callee_calc.RUNS.clear()
    yield
    _callee_rules.THRESHOLD, _callee_rules.UNUSED_SETTING = before
    _callee_calc.c.backend.clear()
    _callee_calc.RUNS.clear()


def _oracle(n, threshold):
    return sum(i for i in range(n) if (i % 100) < threshold)


def test_a_callees_global_invalidates_the_caller():
    """The report: change the constant, get the old answer back."""
    assert _callee_calc.outer(1000) == _oracle(1000, 20)
    _callee_calc.RUNS.clear()

    _callee_rules.THRESHOLD = 60
    got = _callee_calc.outer(1000)

    assert _callee_calc.RUNS, (
        "the caller served a cached answer computed under THRESHOLD=20"
    )
    assert got == _oracle(1000, 60)


def test_the_caller_and_the_callee_agree_within_one_run():
    """The symptom that makes this hard to believe when you hit it.

    The callee's own key always tracked the constant, so a single process
    reported the new rule from one function and the old one from another.
    """
    _callee_calc.outer(1000)
    _callee_rules.THRESHOLD = 60

    assert _callee_calc.inner(1000) == _callee_calc.outer(1000) == _oracle(1000, 60)


def test_two_cached_levels_deep():
    """The chain is transitive, so the fold has to be too."""
    assert _callee_calc.outer_deep(1000) == _oracle(1000, 20)
    _callee_rules.THRESHOLD = 35
    assert _callee_calc.outer_deep(1000) == _oracle(1000, 35)


def test_a_callee_reading_the_global_directly():
    """Without the plain helper in between -- the callee reads it itself."""
    assert _callee_calc.outer_direct(1000) == _oracle(1000, 20)
    _callee_rules.THRESHOLD = 60
    assert _callee_calc.outer_direct(1000) == _oracle(1000, 60)


# --- controls: it must still be a cache -------------------------------------

def test_the_caller_hits_when_nothing_changed():
    """Without this, every assertion above passes on "never cache anything"."""
    assert _callee_calc.outer(1000) == _oracle(1000, 20)
    _callee_calc.RUNS.clear()
    assert _callee_calc.outer(1000) == _oracle(1000, 20)

    assert _callee_calc.RUNS == [], f"the second call recomputed: {_callee_calc.RUNS}"


def test_an_unread_global_does_not_churn_the_key():
    """The fix folds what the callee READS, not everything in its module.

    "Invalidate whenever anything in a dependency's module changes" would pass
    every test above and quietly destroy the hit rate.
    """
    assert _callee_calc.outer(1000) == _oracle(1000, 20)
    _callee_calc.RUNS.clear()

    _callee_rules.UNUSED_SETTING = "loud"
    assert _callee_calc.outer(1000) == _oracle(1000, 20)

    assert _callee_calc.RUNS == [], (
        f"an unread global invalidated the caller: {_callee_calc.RUNS}"
    )


def test_an_unrelated_cached_function_is_unaffected():
    """A function with no path to the rules module keeps hitting."""
    assert _callee_calc.unrelated(7) == 21
    _callee_rules.THRESHOLD = 99
    _callee_calc.RUNS.clear()

    assert _callee_calc.unrelated(7) == 21
    assert _callee_calc.RUNS == []
