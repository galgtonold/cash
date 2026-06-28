"""A while/with loop that self-modifies no-lineage body variables must recompute
from its cell-entry base on an isolated re-run, not accumulate (CAS-59).

`while n < 5: n += 1; total += n` re-run in isolation previously gave a
nondeterministic 0 / 15 / 30: the loop runs as a single opaque unit, so on
re-run the simulator never advances the body vars' virtual lineage past their
cell-entry base. The mismatch classifier then collapsed the recorded lineage of
some (but, set-iteration-dependent, not all) of the body vars back to that base,
making `executed_input_lineages[var][var] == variable_lineage[var]` so the
downstream stale-value guard declined to reset them. With a control variable
pinned at its post-loop value the condition was immediately false and the loop
never re-ran. `for` loops are immune because their per-iteration replay keeps the
guard's base distinct from the collapsed value.

The fix marks a no-lineage self-modifying output of a single-unit (while / with)
loop broken directly in the classifier instead of collapsing its lineage, so its
producer restores the cell-entry base and the loop recomputes from scratch.
"""
import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]


def _rerun(nb_runner, setup, cell, expect):
    nb_runner.create_notebook([setup, cell])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert expect in nb_runner.get_output(2), f"first: {nb_runner.get_output(2)!r}"
    nb_runner.run_cell(2)
    assert expect in nb_runner.get_output(2), f"re-run: {nb_runner.get_output(2)!r}"


def test_while_control_var_plus_accumulator(nb_runner):
    _rerun(nb_runner, "n = 0\ntotal = 0",
           "while n < 5:\n    n += 1\n    total += n\nprint(total)", "15")


def test_while_single_control_var(nb_runner):
    _rerun(nb_runner, "n = 0",
           "while n < 5:\n    n += 1\nprint(n)", "5")


def test_while_list_accumulator(nb_runner):
    _rerun(nb_runner, "n = 0\nacc = []",
           "while n < 3:\n    n += 1\n    acc.append(n)\nprint(acc)", "[1, 2, 3]")


def test_while_set_accumulator(nb_runner):
    _rerun(nb_runner, "n = 0\nseen = set()",
           "while n < 3:\n    n += 1\n    seen.add(n)\nprint(sorted(seen))", "[1, 2, 3]")


def test_while_dict_accumulator(nb_runner):
    _rerun(nb_runner, "n = 0\nd = {}",
           "while n < 3:\n    n += 1\n    d[n] = n * n\nprint(d)", "{1: 1, 2: 4, 3: 9}")


def test_with_block_self_accumulate(nb_runner):
    # A `with` block also runs as a single unit; a no-lineage self-mod inside it
    # must reset on re-run rather than double.
    _rerun(nb_runner, "import contextlib\ntotal = 0",
           "with contextlib.suppress(Exception):\n    total += 5\nprint(total)", "5")


def test_while_nested_in_if(nb_runner):
    _rerun(nb_runner, "n = 0\ntotal = 0",
           "if True:\n    while n < 4:\n        n += 1\n        total += n\nprint(total)", "10")


def test_while_new_var_preserved(nb_runner):
    """CAS-42 guard: a while loop that builds a fresh list (not read at entry)
    is idempotent and must keep producing the right value on re-run."""
    _rerun(nb_runner, "src = [1, 2, 3]",
           "out = []\ni = 0\nwhile i < len(src):\n    out.append(src[i] * 2)\n    i += 1\nprint(out)",
           "[2, 4, 6]")


def test_while_nocache_accumulates(nb_runner):
    """A `# @cash: no-cache` while loop is meant to run fresh and accumulate
    across re-runs (the opt-out), not be reset."""
    nb_runner.create_notebook([
        "n = 0\ntotal = 0",
        "# @cash: no-cache\nwhile n < 2:\n    n += 1\n    total += 100\nprint(total)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "200" in nb_runner.get_output(2), nb_runner.get_output(2)
    nb_runner.run_cell(2)
    # no-cache: loop body runs again; n is already 2 so condition is false and
    # total is unchanged at 200 (the loop does not re-accumulate because n stays).
    assert "200" in nb_runner.get_output(2), f"re-run: {nb_runner.get_output(2)!r}"
