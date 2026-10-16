"""A loop body that changes its loop variable in place before work cash
caches per pass warns KEY-LOOP-VAR-CHANGED, once, naming the lines.

Each pass is keyed on the loop variable as the ``for`` binds it, so the
change is not in the key of the work after it.
"""

import ast
import warnings

import pytest

from tests._cell_driver import run_cash_cell


def _change(src: str, targets=("q",)):
    from cash.notebook.control_structures.loop_target_change import target_changed_before_cached_work

    node = ast.parse(src).body[0]
    return target_changed_before_cached_work(node, set(targets))


@pytest.mark.parametrize(
    "change",
    ["q.append(len(acc))", "q[0] = len(acc)", "q += [len(acc)]", "q.n = len(acc)"],
)
def test_a_change_of_the_loop_variable_before_a_cached_call_is_found(change):
    found = _change(f"for q in items:\n    {change}\n    y = pull(handle)\n")
    assert found is not None
    assert (found.name, found.change_line, found.work_line) == ("q", 2, 3)


@pytest.mark.parametrize(
    "src",
    [
        "for x in xs:\n    y = f(x)\n",
        "for x in xs:\n    acc.append(x)\n    y = f(x)\n",
        "for q in xs:\n    q = q + [1]\n    y = f(q)\n",
        "for q in xs:\n    y = f(q)\n    q.append(1)\n",
        "for q in xs:\n    q.append(1)\n    acc.append(q)\n",
        "for q in xs:\n    q.append(1)\n    print(len(q))\n",
    ],
)
def test_an_ordinary_loop_is_not_flagged(src):
    assert _change(src) is None


def test_cached_work_nested_in_a_branch_counts():
    found = _change("for q in xs:\n    q.append(1)\n    if q:\n        y = f(q)\n")
    assert found is not None and found.work_line == 4


def test_a_binding_that_reads_the_changed_variable_counts():
    assert _change("for q in xs:\n    q.append(1)\n    y = q[0]\n") is not None


def _loop_var_warnings(caught):
    return [w for w in caught if getattr(w.message, "code", None) == "KEY-LOOP-VAR-CHANGED"]


CELL = "accm = []\nfor q in [[1], [1]]:\n    q.append(len(accm))\n    y = slow(q)\n    accm.append(y)\n"


def test_the_cell_warns_once_naming_the_lines(cash_magics):
    run_cash_cell(cash_magics, "import time\ndef slow(q):\n    time.sleep(0.01)\n    return list(q)")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, CELL)
        run_cash_cell(cash_magics, CELL)
    found = _loop_var_warnings(caught)
    assert len(found) == 1, [str(w.message) for w in caught]
    text = str(found[0].message)
    assert "'q'" in text and "line 3" in text and "line 4" in text
    assert "# @cash:no-cache" in text


def test_no_cache_on_the_work_keeps_it_quiet(cash_magics):
    """Positive control first: the same loop without the annotation warns."""
    run_cash_cell(cash_magics, "def slow(q):\n    return list(q)")
    cell = CELL.replace("    y = slow(q)\n", "    # @cash:no-cache\n    y = slow(q)\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, cell)
        quiet = not _loop_var_warnings(caught)
        run_cash_cell(cash_magics, CELL)
    assert quiet
    assert _loop_var_warnings(caught)


def test_an_ordinary_loop_cell_is_quiet(cash_magics):
    """Positive control after: a loop changing its variable warns."""
    run_cash_cell(cash_magics, "def slow(q):\n    return list(q)")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, "acc = []\nfor x in [[1], [2]]:\n    y = slow(x)\n    acc.append(y)\n")
        quiet = not _loop_var_warnings(caught)
        run_cash_cell(cash_magics, CELL)
    assert quiet
    assert _loop_var_warnings(caught)
