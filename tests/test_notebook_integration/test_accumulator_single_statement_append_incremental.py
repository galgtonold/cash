"""Append to a single-statement accumulator loop; count real executions
against a cash-off oracle (CAS-259).

Regression guard for the removed ``cacheable_accumulator_loop`` fast path
(``control_structures/processor.py``, deleted with CAS-259). That mechanism
matched the NARROW shape ``out = []`` immediately followed, in the SAME
cell with no intervening statement, by ``for t in xs: out.append(compute(t))``
-- and routed the WHOLE loop through the statement cache as ONE unit. The
unit's cache key includes the iterable's lineage, so appending a single item
to ``xs`` invalidated the whole unit and re-ran EVERY ``compute()`` call --
exactly as expensive as no caching at all (CAS-259's measured "Arm A").

No existing test caught this. Every prior test of this shape either:

- used a two-statement body (``v = compute(t)`` / ``out.append(v)``), which
  never matched the shape gate at all (``test_call_unit_acceptance.py``
  deliberately avoids the one-statement body for this exact reason), or
- put the ``out = []`` seed in a DIFFERENT cell from the loop
  (``test_cache_calls_directive.py::test_append_loop_caches_the_call_with_no_directive``),
  which also never matched -- the gate only fired when the seed was the
  loop's IMMEDIATELY PRECEDING sibling in the SAME cell, or
- only asserted an UNCHANGED rerun costs zero calls, which both the old
  whole-unit path and the new per-iteration path satisfy identically, so it
  cannot distinguish between them.

This test deliberately uses the exact shape the removed gate matched --
same cell, no intervening statement, single-statement ``Expr(Call)`` body --
and appends one item. Real executions are counted from an external file (not
the kernel's own printed bookkeeping), so cash's own instrumentation cannot
make the assertion vacuous, and compared against a cash-off oracle run with
the IDENTICAL cell shape so a smaller count is provably cash's incremental
reuse, not an artifact of the shape itself.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.loops]

SETUP_ON = "import cash\n%cash_on\nimport time"
SETUP_OFF = "import cash\nimport time"


def _n(path):
    return len(path.read_bytes()) if path.exists() else 0


def _compute_def(counter):
    return (
        "def compute(v):\n"
        f"    open(r'{counter}', 'a').write('X')\n"
        "    time.sleep(0.03)\n"
        "    return v * 10"
    )


def _fast_path_cell(items):
    # `out = []` is the IMMEDIATELY preceding sibling in the SAME cell, and
    # the body is exactly one bare accumulator-method call -- the narrow
    # shape the removed `cacheable_accumulator_loop` gate matched. Do NOT
    # hoist `out = []` into an earlier cell (that would dodge the shape
    # entirely, as the pre-existing tests above did) and do NOT put anything
    # between `out = []` and the `for` (that would also dodge the gate).
    return f"out = []\nfor t in {items}:\n    out.append(compute(t))\nprint('OUT', out)"


def test_single_statement_accumulator_append_is_incremental(nb_runner, tmp_path):
    """Cash on: appending one item to the narrow accumulator-loop shape must
    cost exactly ONE real call, matching per-iteration reuse -- not
    ``len(items)`` calls, which is what the whole-loop unit produced."""
    counter = tmp_path / "calls_on.log"
    nb_runner.create_notebook([SETUP_ON, _compute_def(counter), _fast_path_cell([1, 2])])
    nb_runner.start_kernel()
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 2, f"baseline did not run both iterations: {cold} calls"
    assert "OUT [10, 20]" in nb_runner.get_output(3)

    nb_runner.set_cell_source(3, _fast_path_cell([1, 2, 3]))
    nb_runner.run_cell(3)
    after_append = _n(counter) - cold
    assert after_append == 1, (
        f"appending one item re-ran {after_append} calls, expected 1 (only "
        "the new item) -- the whole loop was re-computed as a single unit"
    )
    assert "OUT [10, 20, 30]" in nb_runner.get_output(3)


def test_single_statement_accumulator_append_oracle_reruns_everything(nb_runner, tmp_path):
    """Cash off, IDENTICAL shape: proves the "1 call" result above is cash's
    incremental reuse and not some property of the shape that would hold
    even with no caching at all."""
    counter = tmp_path / "calls_off.log"
    nb_runner.create_notebook([SETUP_OFF, _compute_def(counter), _fast_path_cell([1, 2])])
    nb_runner.start_kernel(with_cash=False)
    nb_runner.run_all()
    cold = _n(counter)
    assert cold == 2, f"oracle baseline did not run both iterations: {cold} calls"

    nb_runner.set_cell_source(3, _fast_path_cell([1, 2, 3]))
    nb_runner.run_cell(3)
    after_append = _n(counter) - cold
    assert after_append == 3, (
        f"oracle only re-ran {after_append} calls after appending one item; "
        "expected all 3 (no caching at all) -- oracle setup is broken"
    )
