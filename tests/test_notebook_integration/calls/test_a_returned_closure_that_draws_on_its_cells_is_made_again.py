"""A call returning a function that draws on what its closure holds runs
again inside every Run All, as plain Python runs it.

``def mk(): c = itertools.count(); return (lambda: next(c),)``, then ``d =
mk()`` and ``a = (d[0](), d[0]())``: ``next(c)`` changes nothing the shared
reader names, so the result was stored, and the second Run All was handed
the very closure the first had drawn on: ``(2, 3)`` where plain Python gives
``(0, 1)``. A cell holding an iterator, generator or stream, or one the
function changes through ``next`` or ``heapq``, now keeps the result out of
the cache, as a counter's ``nonlocal`` does.
"""

from __future__ import annotations

import pytest

from cash.notebook.call_key import holds_a_closure_with_state
from tests._cell_driver import run_cash_cell
from tools.plain_checker import check, notebook

MAKERS = {
    "count_in_a_tuple": ("c = itertools.count()", "(lambda: next(c),)", "d[0]", (0, 1)),
    "count_bare": ("c = itertools.count()", "lambda: next(c)", "d", (0, 1)),
    "generator": ("c = (i for i in range(10))", "lambda: next(c)", "d", (0, 1)),
    "heappush": ("h = []", "lambda: (heapq.heappush(h, 1), len(h))[1]", "d", (1, 2)),
}


@pytest.mark.parametrize("shape", list(MAKERS))
def test_each_run_all_gets_a_fresh_closure(cash_magics, shape):
    state, returned, draw, expected = MAKERS[shape]
    cells = [
        f"import heapq, itertools, time\ndef mk():\n    time.sleep(0.05)\n    {state}\n    return {returned}",
        "d = mk()",
        f"a = ({draw}(), {draw}())",
    ]
    ns = cash_magics.shell.user_ns
    for run in range(3):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        assert ns["a"] == expected, f"Run All {run + 1}"


def test_a_closure_that_only_reads_its_cells_keeps_no_state():
    """Control: ``lambda x: x * scale['k']`` over a dict it only reads."""
    scale = {"k": 2}
    assert not holds_a_closure_with_state((lambda x: x * scale["k"],))


#: ``(what mk holds, what it returns)``: a closure set on a library module
#: (``textwrap.t = (mk(),)``) drawing through what its cell holds.
ON_A_MODULE = {
    "a_list_of_iterators": ("its = [itertools.count()]", "lambda: next(its[0])"),
    "a_dict_of_iterators": ("d = {'it': itertools.count()}", "lambda: next(d['it'])"),
    "an_object_holding_one": ("b = types.SimpleNamespace(it=itertools.count())", "lambda: next(b.it)"),
    "a_random_of_its_own": ("r = random.Random(0)", "lambda: r.randrange(1000)"),
    "a_deque_popleft": ("dq = collections.deque(range(100))", "lambda: dq.popleft()"),
}


@pytest.mark.integration
@pytest.mark.fresh_kernel
@pytest.mark.timeout(600)
@pytest.mark.parametrize("shape", list(ON_A_MODULE))
def test_a_closure_a_library_module_holds_draws_again(tmp_path, shape):
    """Served from the call cache, the draw handed back the value the first
    Run All drew, and the next draw read the closure moved on once only."""
    state, returned = ON_A_MODULE[shape]
    cells = [
        "import collections, itertools, random, textwrap, time, types",
        f"def mk():\n    {state}\n    return {returned}\ntextwrap.t = (mk(),)",
        "a = (time.sleep(0.05), textwrap.t[0]())[1]",
        "b = (time.sleep(0.3), textwrap.t[0]())[1]",
        "print('R', a, b)",
    ]
    check(notebook(f"closure-on-a-module-{shape}", cells, names=["a", "b"]), tmp_path, ["floors0"]).assert_ok()


@pytest.mark.integration
@pytest.mark.fresh_kernel
@pytest.mark.timeout(600)
def test_a_closure_over_a_count_in_a_tuple_draws_again(tmp_path):
    """``mk()`` returning a function over its own ``itertools.count()``, held
    in a tuple on ``textwrap``: the call ``mk()`` was stored, and the
    second Run All was handed the
    closure the first had drawn on: ``2, 3`` where plain Python draws
    ``0, 1``. Kept out of the cache as the closures above are."""
    cells = [
        "import time, itertools, functools, collections\nimport numpy as np\nimport pandas as pd\nimport textwrap",
        "def mk():\n    it = itertools.count()\n    def f():\n        return next(it)\n    return f\ntextwrap.t = (mk(),)",
        "a = (time.sleep(0.05), textwrap.t[0]())[1]",
        "b = (time.sleep(0.3), textwrap.t[0]())[1]",
        "print('R', a, b)",
    ]
    check(notebook("closure-over-a-count-in-a-tuple", cells, names=["a", "b"]), tmp_path, ["floors0"]).assert_ok()
