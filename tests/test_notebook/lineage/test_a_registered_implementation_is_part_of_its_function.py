"""A ``functools.singledispatch`` function is keyed with every implementation it dispatches to.

Keyed by its base function alone, editing ``@fmt.register def _(x: int)``
and running the notebook again served the result of the old implementation,
whether the implementation was written in a cell or in a local module.
"""

import os
import sys
import time

import pytest

from cash.tracking.module_symbols import closure_digest_of_source
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

DISPATCH = (
    "from functools import singledispatch\n\n"
    "@singledispatch\ndef fmt(x):\n    return 'obj'\n\n"
    "@fmt.register\ndef _(x: int):\n    return x + {n}\n"
)


def test_editing_an_implementation_in_a_cell_recomputes(cash_magics, mock_shell):
    compute = f"import time\ndef compute(x):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return fmt(x)"
    cells = [DISPATCH.format(n=1), compute, "r = compute(3)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert mock_shell.user_ns["r"] == 4

    cells[0] = DISPATCH.format(n=100)
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    assert mock_shell.user_ns["r"] == 103


def test_editing_a_registration_type_changes_the_function_digest(statement_processor):
    from functools import singledispatch

    def build(kind):
        @singledispatch
        def show(x):
            return "obj"

        def impl(x):
            return "special"

        show.register(kind, impl)
        return show

    tracker = statement_processor.function_tracker
    assert tracker.get_function_source_hash(build(int)) != tracker.get_function_source_hash(build(str))


def test_a_registration_is_in_the_source_of_the_name_it_registers_on():
    before = closure_digest_of_source(DISPATCH.format(n=1), {"fmt"})
    after = closure_digest_of_source(DISPATCH.format(n=100), {"fmt"})
    assert before is not None and after is not None
    assert before != after


@pytest.fixture
def dispatch_module(tmp_path, monkeypatch):
    name = f"_dispatch_mod_{os.getpid()}_{id(tmp_path)}"
    path = tmp_path / f"{name}.py"
    path.write_text(DISPATCH.format(n=1), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name, path
    sys.modules.pop(name, None)


def test_editing_an_implementation_in_a_module_recomputes(cash_magics, mock_shell, dispatch_module):
    name, path = dispatch_module
    compute = (
        f"import {name}, time\ndef compute(x):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return {name}.fmt(x)"
    )
    cells = [compute, "r = compute(3)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert mock_shell.user_ns["r"] == 4

    before = os.stat(path).st_mtime_ns
    path.write_text(DISPATCH.format(n=100), encoding="utf-8")
    os.utime(path, ns=(before + 2_000_000_000, before + 2_000_000_000))
    started = time.perf_counter()
    run_cash_cell(cash_magics, cells[1], cells=cells)

    assert mock_shell.user_ns["r"] == 103
    assert time.perf_counter() - started >= ABOVE_PERSISTENCE_FLOOR_S
