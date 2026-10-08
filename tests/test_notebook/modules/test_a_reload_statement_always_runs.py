"""``importlib.reload(mylib)`` in a cell runs on every run of the notebook.

``mylib.K = 5`` in one cell, ``importlib.reload(mylib)`` in the next: a
module that takes a moment to import made the reload slow enough to store,
and the second run served it from the cache. The module kept ``K = 5`` and
the reader below computed with it, where a plain kernel has the file's
``K = 1``. A reload sets every global of the module anew, so it is a write
of the module's state: never stored, and the module's lineage moves.
"""

from __future__ import annotations

import os
import sys

import pytest

from cash.notebook.callee_reach import module_state_writes
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

LIB = f"import time\ntime.sleep({ABOVE_PERSISTENCE_FLOOR_S})\nK = 1\ndef from_k(x):\n    return K * x\n"


@pytest.fixture
def lib(tmp_path, monkeypatch):
    name = f"_reload_lib_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(LIB, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name
    sys.modules.pop(name, None)


@pytest.mark.parametrize(
    ("imports", "reload"),
    [
        ("import {m}, importlib", "importlib.reload({m})"),
        ("import {m}\nfrom importlib import reload", "reload({m})"),
    ],
    ids=["importlib_reload", "reload_imported_by_name"],
)
def test_the_second_run_reloads_again(cash_magics, lib, imports, reload):
    cells = [imports.format(m=lib), f"{lib}.K = 5", reload.format(m=lib), f"x = {lib}.from_k(2)"]
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)

    assert sys.modules[lib].K == 1
    assert cash_magics.shell.user_ns["x"] == 2


def test_a_reload_is_a_write_of_the_module_s_state(cash_magics, lib):
    run_cash_cell(cash_magics, f"import {lib}, importlib")
    ns = cash_magics.shell.user_ns

    assert module_state_writes(f"importlib.reload({lib})", ns) == {lib}


def test_a_method_named_reload_is_not_a_reload(cash_magics, lib):
    """Positive control: only ``importlib.reload`` counts, not any
    ``.reload(...)`` call that happens to take the module."""
    run_cash_cell(cash_magics, f"import {lib}\nclass Cfg:\n    def reload(self, m):\n        return m\ncfg = Cfg()")
    ns = cash_magics.shell.user_ns

    assert module_state_writes(f"cfg.reload({lib})", ns) == frozenset()
