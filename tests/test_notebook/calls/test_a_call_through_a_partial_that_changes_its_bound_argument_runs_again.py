"""A call through a ``functools.partial`` that changes the object the partial
holds runs again on a re-run.

``fill = partial(add, bucket)`` with ``def add(store, x): store.append(x)``:
each ``fill(i)`` appends to ``bucket``, which the statement never names. A
call slow enough to be stored was kept with no record of that change, and
served on the cell's re-run without it: a busy machine printed ``[0, 2]``
where Python prints ``[0, 1, 2]``. The objects a partial hands its function
in the parameters that function changes are now changed by the call.
"""

from __future__ import annotations

import ast
import functools
import linecache

import pytest

from cash.analysis.handed_callables import handed_callables
from tests._cell_driver import run_cash_cell

SETUP = (
    "import time\n"
    "from functools import partial\n"
    "bucket = []\n"
    "def add(store, x):\n"
    "    if x == 1:\n"
    "        time.sleep(0.05)  # over the time a statement must take to be stored\n"
    "    store.append(x)"
)

CELLS = {
    "in_a_loop": "fill = partial(add, bucket)\nfor i in range(3):\n    fill(i)\nprint(bucket)",
    "one_call_a_line": "fill = partial(add, bucket)\nfill(0)\nfill(1)\nfill(2)\nprint(bucket)",
    "by_keyword": "fill = partial(add, store=bucket)\nfor i in range(3):\n    fill(x=i)\nprint(bucket)",
    "partial_of_a_partial": "fill = partial(partial(add), bucket)\nfor i in range(3):\n    fill(i)\nprint(bucket)",
}


@pytest.mark.parametrize("cell", CELLS.values(), ids=CELLS.keys())
def test_the_rerun_appends_every_item(cash_magics, cell):
    ns = cash_magics.shell.user_ns
    for _ in range(2):
        run_cash_cell(cash_magics, SETUP)
        run_cash_cell(cash_magics, cell)
        assert ns["bucket"] == [0, 1, 2], ns["bucket"]
        assert ns["fill"].args[:1] == (ns["bucket"],) or ns["fill"].keywords.get("store") is ns["bucket"]


def _defined(source, name):
    """A namespace holding the function *source* defines, with its source readable."""
    filename = f"<partial-test-{name}>"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = {}
    exec(compile(source, filename, "exec"), namespace)
    return namespace


def _receivers(code, namespace):
    return handed_callables(ast.parse(code), namespace, scope="no_control_bodies").receivers


def test_the_call_changes_the_object_the_partial_holds():
    namespace = _defined("def add(store, x):\n    store.append(x)\n", "add")
    namespace["bucket"] = []
    namespace["fill"] = functools.partial(namespace["add"], namespace["bucket"])
    assert {"bucket", "fill"} <= _receivers("fill(1)", namespace)


def test_a_bound_argument_the_function_only_reads_is_not_changed():
    namespace = _defined("def total(store, x):\n    return sum(store) + x\n", "total")
    namespace["bucket"] = [1, 2]
    namespace["count"] = functools.partial(namespace["total"], namespace["bucket"])
    assert "bucket" not in _receivers("n = count(1)", namespace)


def test_a_builtin_method_that_changes_its_object_changes_it():
    namespace = {"bucket": []}
    namespace["fill"] = functools.partial(list.append, namespace["bucket"])
    assert "bucket" in _receivers("fill(1)", namespace)
