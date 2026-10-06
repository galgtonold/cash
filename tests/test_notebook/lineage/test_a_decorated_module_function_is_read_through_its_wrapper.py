"""A statement calling a decorated function of a local module is keyed on what that function reads.

``x = mylib.dk(2)`` with ``dk`` under ``@cash.cache`` (or the user's own
``functools.wraps`` decorator) reading ``mylib.K`` or ``os.environ``: the
walk stopped at the wrapper -- ``@cash.cache``'s lives in cash, the user's
reaches the function only through its closure -- so the statement was keyed
on neither, and editing ``mylib.K = 5`` to ``7`` served the old result, even
after Restart & Run All.
"""

import os
import sys

import pytest

from cash.notebook.callee_reach import reached_user_code
from cash.notebook.lineage_formula import statement_environment_reads
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

LIB = (
    "import functools, os, time\nimport cash\nK = 1\n"
    "def timed(f):\n    @functools.wraps(f)\n    def w(*a, **k):\n        return f(*a, **k)\n    return w\n"
    "def bare(f):\n    def w(*a, **k):\n        return f(*a, **k)\n    return w\n"
    "@cash.cache\ndef cached_k(n):\n    return K * n\n"
    "@cash.cache\ndef cached_mode():\n    return os.environ.get('CASH_UT_DEC_MODE', '-')\n"
    f"@timed\ndef timed_k(n):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return K * n\n"
    "@bare\ndef bare_mode():\n    return os.environ.get('CASH_UT_DEC_MODE', '-')\n"
)


@pytest.fixture
def lib(tmp_path, monkeypatch):
    name = f"_declib_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(LIB, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("CASH_UT_DEC_MODE", raising=False)
    yield name
    sys.modules.pop(name, None)


@pytest.mark.parametrize("fn", ["cached_k", "timed_k"])
def test_the_module_data_the_wrapped_function_reads_is_reached(lib, fn):
    module = __import__(lib)
    reach = reached_user_code(f"x = lib.{fn}(2)", {"lib": module})
    assert f"{lib}.K" in dict(reach.data)


@pytest.mark.parametrize("fn", ["cached_mode", "bare_mode"])
def test_the_environment_the_wrapped_function_reads_is_reached(lib, fn):
    module = __import__(lib)
    assert ("env", "CASH_UT_DEC_MODE") in statement_environment_reads(f"x = lib.{fn}()", {"lib": module})


def test_editing_the_setting_and_running_again_recomputes(cash_magics, mock_shell, lib):
    cells = [f"import {lib}", f"{lib}.K = 5", f"x = {lib}.timed_k(2)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert mock_shell.user_ns["x"] == 10

    cells[1] = f"{lib}.K = 7"
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert mock_shell.user_ns["x"] == 14
