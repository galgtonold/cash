"""End-to-end sub-expression caching, default-on with an opt-out (CAS-243).

The two shapes the feature exists for, both of which statement-level caching
cannot help with:

- ``out.append(compute(t))`` — skip-cached because the append is a mutation, so
  without interception it re-runs in full on every execution.
- ``s += compute(x)`` — cached, but keyed on the running prefix, so reordering
  the iterable re-runs everything after the first change.

Interception used to be opt-in via ``# @cash:cache-calls``; task 10 flips the
default, so both shapes above are cached with NO directive at all, and
``# @cash:no-cache-calls`` is the escape hatch that turns it back off. The
expensive call is cached and the cheap wrapper still executes, which is both
faster *and* more correct than skip-caching: the mutation genuinely happens
every run.

Counted, never timed — wall-clock cannot distinguish "recomputed" from
"restored but slow".
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.loops]

# Above the cost model's floor, or nothing is stored and every assertion below
# holds whether or not the feature works.
_SLEEP = 0.2


def _helpers(log):
    return (
        "import time, pathlib\n"
        f"LOG = pathlib.Path(r'{log}')\n"
        "def compute(x):\n"
        "    with LOG.open('a') as fh:\n"
        "        fh.write(str(x) + '\\n')\n"
        f"    time.sleep({_SLEEP})\n"
        "    return x + 1\n"
    )


def _n(log):
    return len(log.read_text().splitlines()) if log.exists() else 0


def test_append_loop_caches_the_call_with_no_directive(nb_runner, tmp_path):
    """No directive anywhere -- interception is on by default."""
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([
        _helpers(log),
        "out = []",
        "for t in [1, 2]:\n    out.append(compute(t))\nprint('OUT', out)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _n(log) == 2, "baseline did not run both iterations"
    assert "OUT [2, 3]" in nb_runner.get_output(3)

    nb_runner.run_cell(3)
    assert _n(log) == 2, (
        "compute() re-ran; call caching should be on by default, no directive needed"
    )
    assert "OUT [2, 3]" in nb_runner.get_output(3), (
        "the append stopped executing -- the mutation must still happen every run"
    )


def test_no_cache_calls_directive_turns_interception_off(nb_runner, tmp_path):
    """Positive control: the opt-out must actually opt out.

    Without this, a bug that made ``# @cash:no-cache-calls`` a no-op would
    still pass the default-on test above -- interception would just stay on
    regardless of what the directive says.

    Note where the directive sits: ``@cash:`` directives attach to the
    statement *below* them, so it has to be on the loop header, not indented
    inside the body -- placed inside the body it would attach to the
    ``out.append(...)`` statement itself, which is exactly what we want here,
    but placed above ``out = []`` instead it would scope to the wrong
    statement and silently measure an intercepted (cached) run. Putting it on
    the ``for`` line is what actually reaches the statement doing the call.
    """
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([
        _helpers(log),
        "out = []",
        "# @cash:no-cache-calls\nfor t in [1, 2]:\n    out.append(compute(t))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _n(log) == 2

    nb_runner.run_cell(3)
    assert _n(log) == 4, "no-cache-calls did not disable interception"


def test_no_cache_calls_from_the_cell_header_covers_every_statement(nb_runner, tmp_path):
    """The opt-out is cell-scoped too: a directive in the cell's LEADING
    comment block must reach every top-level statement in the cell, not just
    the first one -- exactly the placement trap default-on introduces (see
    the module docstring). Two independent ``append`` statements, each with
    its own compute call, both must stay uncached.

    Deliberately two plain top-level statements rather than a second ``for``
    loop: a control structure's header is resolved by its OWN local scan
    (``resolve_header_annotation`` in ``control_structures/helpers.py``),
    separate from the cell-header mechanism (``leading_cell_annotation``)
    this test targets, and that local scan stops at the first line of code
    above it -- it does not see past an intervening statement like ``out =
    []``. Plain top-level statements are what ``leading_cell_annotation``
    actually guarantees reaches every one of, per its own docstring.
    """
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([
        _helpers(log),
        (
            "# @cash:no-cache-calls\n"
            "out = []\n"
            "out.append(compute(1))\n"
            "out.append(compute(2))"
        ),
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _n(log) == 2, "baseline did not run both calls"

    nb_runner.run_cell(2)
    assert _n(log) == 4, (
        "the header opt-out did not reach the second append statement -- it "
        "only covered the first"
    )


def test_badge_names_the_mechanism_that_cached_the_call(nb_runner, tmp_path):
    """The badge must still say where the cache came from, even with no
    directive written -- interception is silent by default, so the badge is
    the only place a reader can confirm it engaged.
    """
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([
        "%cash_badge print",
        _helpers(log),
        "out = []",
        "for t in [1, 2]:\n    out.append(compute(t))",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    badge = nb_runner.get_output(4)

    assert "compute()" in badge, badge
    assert "cache-calls" in badge, (
        f"the badge does not say the call was cached by interception:\n{badge}"
    )


def test_reordering_an_accumulator_fold_costs_nothing_with_no_directive(nb_runner, tmp_path):
    """The user-reported case: a reorder must stop re-running the tail, with
    no directive needed.
    """
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([
        _helpers(log),
        "s = 0\nfor x in [1, 10, 5]:\n    s += compute(x)\nprint('SUM', s)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _n(log) == 3
    assert "SUM 19" in nb_runner.get_output(2)

    nb_runner.set_cell_source(
        2, "s = 0\nfor x in [5, 10, 1]:\n    s += compute(x)\nprint('SUM', s)"
    )
    nb_runner.run_cell(2)
    assert "SUM 19" in nb_runner.get_output(2)
    assert _n(log) == 3, (
        "a reorder re-ran compute(); the call cache is order-independent by "
        "construction and must absorb it, with no directive needed"
    )

    # A genuinely new value costs exactly one call, and no more.
    nb_runner.set_cell_source(
        2, "s = 0\nfor x in [5, 10, 1, 7]:\n    s += compute(x)\nprint('SUM', s)"
    )
    nb_runner.run_cell(2)
    assert "SUM 27" in nb_runner.get_output(2)
    assert _n(log) == 4, "adding one item cost more than one call"


def test_no_cache_calls_disables_the_reordering_win_too(nb_runner, tmp_path):
    """Positive control for the reorder case: with the opt-out, a reorder
    must re-run the tail exactly like it did before this feature existed --
    proving the opt-out actually disabled interception here too, not just for
    the append shape above.
    """
    log = tmp_path / "calls.log"
    nb_runner.create_notebook([
        _helpers(log),
        "s = 0\n# @cash:no-cache-calls\nfor x in [1, 10, 5]:\n    s += compute(x)\nprint('SUM', s)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _n(log) == 3
    assert "SUM 19" in nb_runner.get_output(2)

    nb_runner.set_cell_source(
        2, "s = 0\n# @cash:no-cache-calls\nfor x in [5, 10, 1]:\n    s += compute(x)\nprint('SUM', s)"
    )
    nb_runner.run_cell(2)
    assert "SUM 19" in nb_runner.get_output(2)
    assert _n(log) == 6, (
        "no-cache-calls should have disabled the call cache, so the reorder "
        "re-runs compute() for every element again"
    )
