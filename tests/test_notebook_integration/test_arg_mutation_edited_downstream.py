"""Editing a function that mutates its argument must invalidate a CACHED
downstream reader of the mutated variable.

A bare ``proc(data)`` call mutates its argument in place (return discarded); a
later, expensive-enough-to-cache statement reads the mutated object. Editing
``proc`` must recompute that reader, not serve its stale cached value.

Before the fix the bare call surfaced NO output, so the mutated variable's
lineage was never bumped -- it stayed a constant derivation token, and the
cached downstream reader keyed on it hit stale (a silent wrong value). This
reproduced for dict / list / DataFrame alike; a DataFrame only surfaced it first
because ``.sum()`` is naturally expensive enough to cache. The separate-cell
variant also exercises the cross-cell restore: the mutated object must be
restored, not its pre-mutation constructor.

Run through ``scripts/fails_first.py`` to confirm these fail without the fix.
"""
import pytest

pytestmark = pytest.mark.core

# ~1.2M iterations -> comfortably above the 10ms cache floor, so the reader caches.
_READER = "result = sum(v for v in data for _ in range(3)) + data[-1]"


def _num(out: str) -> str:
    for line in out.splitlines():
        if line.startswith("RESULT"):
            return line.split(None, 1)[1].strip()
    return f"<no RESULT: {out[:60]!r}>"


def test_arg_mutation_edit_invalidates_cached_reader_same_cell(nb_runner):
    nb_runner.create_notebook([
        "pass",
        "def proc(x):\n    x.append(1)",
        f"data = list(range(400000))\nproc(data)\n{_READER}",
        "print('RESULT', result)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    v1 = _num(nb_runner.get_output(4))

    nb_runner.set_cell_source(2, "def proc(x):\n    x.append(9999)")
    nb_runner.run_cells([2, 3, 4])
    v2 = _num(nb_runner.get_output(4))

    assert v2.isdigit(), f"reader errored or produced no RESULT: {v2!r}"
    assert v1 != v2, f"stale cached reader: edit to proc not reflected ({v1} == {v2})"


def test_arg_mutation_edit_invalidates_cached_reader_separate_cell(nb_runner):
    # Reader lives in a LATER cell -> also exercises cross-cell restore of the
    # mutated object (must restore the mutation, not the pre-mutation ctor).
    nb_runner.create_notebook([
        "pass",
        "def proc(x):\n    x.append(1)",
        "data = list(range(400000))\nproc(data)",
        f"{_READER}\nprint('RESULT', result)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    v1 = _num(nb_runner.get_output(4))

    nb_runner.set_cell_source(2, "def proc(x):\n    x.append(9999)")
    nb_runner.run_cells([2, 3, 4])
    v2 = _num(nb_runner.get_output(4))

    assert v2.isdigit(), f"reader errored or produced no RESULT (cross-cell revert?): {v2!r}"
    assert v1 != v2, f"stale cached reader: edit to proc not reflected ({v1} == {v2})"


def test_arg_mutation_no_edit_still_hits(nb_runner):
    """Guard against over-invalidation: with proc UNCHANGED, re-running the reader
    must serve the same value (the fix must not force a spurious recompute)."""
    nb_runner.create_notebook([
        "pass",
        "def proc(x):\n    x.append(1)",
        "data = list(range(400000))\nproc(data)",
        f"{_READER}\nprint('RESULT', result)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    v1 = _num(nb_runner.get_output(4))
    nb_runner.run_cells([4])          # re-run the reader only, no edit
    v2 = _num(nb_runner.get_output(4))
    assert v1 == v2 and v1.isdigit(), f"unexpected change on no-edit re-run ({v1} -> {v2})"
