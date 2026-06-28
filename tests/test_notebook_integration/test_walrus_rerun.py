"""Walrus operator (``:=`` / ast.NamedExpr) correctness under isolated re-run.

A self-referential walrus (``n := n + 1``) reads the old value before binding,
so on an isolated cell re-run cash must restore the cell-entry base rather than
accumulate -- exactly as for ``n = n + 1``. The bug was that ``NamedExpr`` had no
handling in the AST flow analyzer: the target bound before its own value was
read, so the self-read never registered as an input and the no-lineage
self-write guard skipped it. PEP 572 also binds a comprehension walrus in the
enclosing scope, so ``[(t := t + i) for ...]`` exposes a cell-level output.
"""
import pytest

pytestmark = pytest.mark.upstream


def test_walrus_self_accumulate_rerun(nb_runner):
    """``[(total := total + i) for ...]`` re-run restarts from total's base."""
    nb_runner.create_notebook([
        "total = 0",
        "vals = [(total := total + i) for i in range(4)]\nprint(vals, total)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "[0, 1, 3, 6] 6" in nb_runner.get_output(2)
    nb_runner.run_cell(2)
    out = nb_runner.get_output(2)
    assert "[0, 1, 3, 6] 6" in out, f"walrus self-accumulate not reset on re-run: {out!r}"


def test_walrus_simple_rebind_rerun(nb_runner):
    """``if (n := n + 1)`` re-run reproduces the single-run value (no double bump)."""
    nb_runner.create_notebook([
        "n = 10",
        "if (n := n + 1):\n    pass\nprint(n)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "11" in nb_runner.get_output(2)
    nb_runner.run_cell(2)
    out = nb_runner.get_output(2)
    assert out.strip().endswith("11"), f"walrus rebind accumulated on re-run: {out!r}"


def test_walrus_downstream_propagation(nb_runner):
    """Regression guard: a non-self-referential walrus still carries lineage so
    an upstream edit propagates to a consumer of the walrus-assigned name."""
    nb_runner.create_notebook([
        "base = 5",
        "m = (doubled := base * 2)\nprint(m)",
        "out = doubled + 1\nprint(out)",
    ])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "11" in nb_runner.get_output(3)
    nb_runner.set_cell_source(1, "base = 50")
    nb_runner.run_cell(2)
    nb_runner.run_cell(3)
    assert "101" in nb_runner.get_output(3), \
        f"downstream did not see walrus-assigned update: {nb_runner.get_output(3)!r}"
