"""Guard for the loop-reuse claims in ``docs/known-limitations.md``.

Both behaviours below are guarded with ``# @cash:no-cache-calls`` on the loop
header. That is deliberate, not incidental: call-level caching (CAS-243) is
on by DEFAULT now (task 10), and it is precisely the fix for both documented
limitations here -- an intercepted ``compute(x)`` is order-independent, so a
reordered accumulator fold no longer re-runs the tail, and an intercepted
call inside ``out.append(...)`` is reused across an unchanged rerun instead
of always recomputing. Left undirected, this file would no longer be
measuring the limitation it exists to pin; it would be measuring the fix.
The opt-out keeps this file honest about the baseline behaviour a user who
switches interception off (or a call site the gate refuses) still hits.

Two documented behaviours, both easy to break and neither covered by a timing
test (wall-clock cannot separate "recomputed" from "restored but slow" — every
assertion here counts calls):

1. **"Reordering a loop's items re-runs the tail"** — a body that folds into a
   running accumulator makes iteration *k*'s key depend on every iteration
   before it. The docs ship a measured table; this pins its shape:

   | change to the list | iterations re-run |
   |---|---|
   | append | just the new one |
   | swap the last two | 2 of 3 |
   | new first element | all 3 |
   | back to a list seen before | none |

   And the contrast that makes it "the fold, not the loop": drop the
   accumulator and reordering is free.

2. **The quickstart's "collect with an assignment or a store, not append"
   warning.** Inside a loop body the two mutation shapes behave differently and
   the difference is not guessable: a subscript store (``prices[t] = f(t)``)
   restores, while ``out.append(f(t))`` re-executes every run. The quickstart
   demonstrates loop caching with the subscript form and warns off ``append``,
   so both halves are pinned here.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.loops]

# Above the cost model's 0.01s min_execution_time_to_cache_seconds, so the
# statement is actually eligible for caching. Without this the whole file
# passes vacuously: nothing is ever stored, so nothing can be reused.
_SLEEP = 0.05


def _helpers(log):
    return f"""
import time, pathlib
LOG = pathlib.Path(r"{log}")

def compute(x):
    with LOG.open("a") as fh:
        fh.write(f"{{x}}\\n")
    time.sleep({_SLEEP})
    return x + 1
"""


def _calls(log):
    """Total recorded compute() calls so far."""
    return len(log.read_text().splitlines()) if log.exists() else 0


def test_accumulator_fold_reuses_only_the_unchanged_prefix(nb_runner, tmp_path):
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([
        _helpers(log),
        "s = 0\n# @cash:no-cache-calls\nfor x in [1, 10]:\n    s += compute(x)\nprint('SUM', s)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "SUM 13" in nb_runner.get_output(2)
    assert _calls(log) == 2, "baseline did not run both iterations"

    def rerun(lst):
        before = _calls(log)
        nb_runner.set_cell_source(
            2, f"s = 0\n# @cash:no-cache-calls\nfor x in {lst}:\n    s += compute(x)\nprint('SUM', s)"
        )
        nb_runner.run_cell(2)
        return _calls(log) - before

    # Appending leaves every earlier prefix intact -> only the new tail runs.
    assert rerun("[1, 10, 5]") == 1, "append re-ran more than the new item"
    assert "SUM 19" in nb_runner.get_output(2)

    # Same first element, different tail: the divergence starts at position 2.
    assert rerun("[1, 5, 10]") == 2, "swapping the tail did not re-run exactly the tail"

    # A different FIRST element makes every prefix new.
    assert rerun("[5, 10, 1]") == 3, "changing the head did not re-run all iterations"

    # A sequence whose every prefix was seen before costs nothing.
    assert rerun("[1, 10, 5]") == 0, "returning to a cached ordering still recomputed"


def test_accumulator_fold_reordering_is_free_by_default(nb_runner, tmp_path):
    """The undirected twin of the test above -- no directive anywhere.

    Matches known-limitations.md's NEW headline table: append still costs
    one call (a genuinely new value), but every reorder after that costs
    ZERO -- call-level caching (on by default, CAS-243) makes ``compute(x)``
    order-independent, even though the *statement*'s own prefix-keyed cache
    entry still misses on every one of these edits. Without this test,
    known-limitations.md's new table and quickstart's rewritten claim that
    reordering is free by default are pinned by nothing (the file only had
    the opt-out variant before).
    """
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([
        _helpers(log),
        "s = 0\nfor x in [1, 10]:\n    s += compute(x)\nprint('SUM', s)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "SUM 13" in nb_runner.get_output(2)
    assert _calls(log) == 2, "baseline did not run both iterations"

    def rerun(lst):
        before = _calls(log)
        nb_runner.set_cell_source(
            2, f"s = 0\nfor x in {lst}:\n    s += compute(x)\nprint('SUM', s)"
        )
        nb_runner.run_cell(2)
        return _calls(log) - before

    # Appending: still one genuinely new value, one call -- same as the
    # opt-out case, since `10` was never a new value either way.
    assert rerun("[1, 10, 5]") == 1, "append cost more than the new item"
    assert "SUM 19" in nb_runner.get_output(2)

    # Swapping the tail: the statement misses (prefix chain broke), but
    # BOTH compute(1) and compute(10) were already called with these exact
    # arguments -- zero new calls, not 2.
    assert rerun("[1, 5, 10]") == 0, "swapping the tail cost calls by default"

    # A different FIRST element: same reasoning -- 5 and 10 are still
    # cached values, only the identity of "first" changed.
    assert rerun("[5, 10, 1]") == 0, "changing the head cost calls by default"

    # A sequence whose every value was seen before costs nothing, same as
    # the opt-out case.
    assert rerun("[1, 10, 5]") == 0, "a fully-seen ordering still recomputed"


def test_without_an_accumulator_reordering_is_free(nb_runner, tmp_path):
    """The contrast that makes the entry honest: it is the fold, not the loop."""
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([
        _helpers(log),
        "for x in [1, 10, 5]:\n    y = compute(x)\nprint('Y', y)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _calls(log) == 3

    nb_runner.set_cell_source(2, "for x in [5, 10, 1]:\n    y = compute(x)\nprint('Y', y)")
    nb_runner.run_cell(2)
    assert _calls(log) == 3, (
        "reordering re-ran iterations even without an accumulator — the "
        "known-limitations entry blames the fold and would now be wrong"
    )


def test_subscript_store_caches_in_a_loop_but_append_does_not(nb_runner, tmp_path):
    """Quickstart's 'collect with an assignment or a store, not append' warning.

    The quickstart's loop example uses the subscript form, so it must restore;
    the warning next to it says ``append`` does not, so that must re-run. Both
    directions matter — asserting only one would let the example rot into the
    shape the warning tells readers to avoid.

    The append loop carries ``# @cash:no-cache-calls``: left undirected, the
    now-default call-level cache would reuse ``compute(t)`` across the rerun
    and the quickstart's warning would stop being true for a reader who
    hasn't opted out of anything.
    """
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([
        _helpers(log),
        "prices = {}\nout = []",
        "for t in [1, 2]:\n    prices[t] = compute(t)",   # store: caches
        "# @cash:no-cache-calls\nfor t in [1, 2]:\n    out.append(compute(t))",   # append: re-executes
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _calls(log) == 4, "baseline did not run both loops"

    before = _calls(log)
    nb_runner.run_cell(3)
    store_calls = _calls(log) - before
    before = _calls(log)
    nb_runner.run_cell(4)
    append_calls = _calls(log) - before

    assert store_calls == 0, (
        f"the subscript-store loop re-ran ({store_calls} calls); it is the "
        "quickstart's loop-caching example and must restore"
    )
    assert append_calls == 2, (
        f"the append loop restored ({append_calls} calls); the quickstart warns "
        "readers off append for exactly this reason and would now be wrong"
    )


def test_subscript_store_and_undirected_append_both_reuse_by_default(nb_runner, tmp_path):
    """The undirected twin of the test above.

    Quickstart's rewritten warning box says a `.append()` loop "still skips
    the slow part" by default even though the append itself keeps
    re-executing. Without a directive anywhere, this pins that: the append
    loop now costs ZERO calls on a re-run, same as the store loop -- the
    only difference between the two shapes left is which one the STATEMENT
    itself restores versus re-executes, not whether `compute()` gets called
    again.
    """
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([
        _helpers(log),
        "prices = {}\nout = []",
        "for t in [1, 2]:\n    prices[t] = compute(t)",   # store: caches
        "for t in [1, 2]:\n    out.append(compute(t))",   # append: still cached by default
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _calls(log) == 4, "baseline did not run both loops"

    before = _calls(log)
    nb_runner.run_cell(3)
    store_calls = _calls(log) - before
    before = _calls(log)
    nb_runner.run_cell(4)
    append_calls = _calls(log) - before

    assert store_calls == 0, (
        f"the subscript-store loop re-ran ({store_calls} calls); it is the "
        "quickstart's loop-caching example and must restore"
    )
    assert append_calls == 0, (
        f"the undirected append loop re-ran compute() ({append_calls} calls); "
        "quickstart claims it is cached by default now and would be wrong"
    )
