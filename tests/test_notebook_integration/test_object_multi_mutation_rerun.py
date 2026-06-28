"""Isolated re-run of an object mutated by a no-output method call (CAS-53).

A lineage-carrying object mutated in place by a bare method call
(``b.items.append(..)``) skips the per-statement cache (no output), so on an
isolated re-run it would accumulate. The CAS-8 method-mutation lineage bump makes
the receiver lineage-carrying, diverting it from the no-lineage accumulation
guard; the fix routes such METHOD receivers back through that guard so the
receiver is restored to its cell-entry base.

Scoped to method receivers, so subscript/attribute in-place writes
(``df['col']=..``) keep their per-statement cache (the CAS-42 design) — covered
by ``test_df_subscript_self_scale_not_over_reset`` below and the voladj suite.
"""
import pytest

pytestmark = pytest.mark.upstream

BOX = ("class Box:\n    def __init__(self, n):\n        self.items = []\n"
       "        self.n = n\nb = Box(0)")


def _rerun(nb_runner, cells, expect):
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert expect in nb_runner.get_output(3), f"first run: {nb_runner.get_output(3)!r}"
    nb_runner.run_cell(2)
    nb_runner.run_cell(3)
    assert expect in nb_runner.get_output(3), f"re-run: {nb_runner.get_output(3)!r}"


def test_method_mutation_plus_attr_increment(nb_runner):
    """b.items.append(b.n) + b.n += 1 must reset to base on isolated re-run."""
    _rerun(nb_runner, [BOX, "b.items.append(b.n)\nb.n += 1", "print(b.items, b.n)"], "[0] 1")


def test_results_append_and_counter(nb_runner):
    """Common real pattern: accumulate a list and a counter on one object."""
    _rerun(nb_runner, [
        "class S:\n    def __init__(self):\n        self.rows = []\n        self.total = 0\ns = S()",
        "s.rows.append(10)\ns.total += 10",
        "print(s.rows, s.total)",
    ], "[10] 10")


def test_single_attr_increment_still_ok(nb_runner):
    _rerun(nb_runner, [BOX, "b.n += 1", "print(b.n)"], "1")


def test_single_method_append_still_ok(nb_runner):
    _rerun(nb_runner, [BOX, "b.items.append(5)", "print(b.items)"], "[5]")


def test_df_subscript_self_scale_not_over_reset(nb_runner):
    """CAS-42 guard: a subscript in-place write is NOT a method receiver, so it
    keeps its per-statement cache and still re-runs idempotently (not doubled)."""
    _rerun(nb_runner, [
        "import pandas as pd\ndf = pd.DataFrame({'a': list(range(1, 9))})",
        "df['a'] = df['a'] * 2",
        "print(df['a'].tolist())",
    ], "[2, 4, 6, 8, 10, 12, 14, 16]")
