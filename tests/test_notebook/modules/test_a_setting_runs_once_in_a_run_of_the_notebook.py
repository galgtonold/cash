"""Running the notebook runs each statement that changes a module's object once.

``mylib.LOG.append(v)`` in two cells, run top to bottom: the second one, and
the cell after it, found the module "changed" above them and ran the appends
before them again, so ``LOG`` held ``[1, 1, 20, 1, 20]``. The runtime keyed
the append with the module as a mere input and the upstream simulation with
the module as an output, which adds the module's lineage to the key; the two
keys differed, so did the lineage the append gave the module.
"""

import os
import sys

import pytest

from tests._cell_driver import run_cash_cell


@pytest.fixture
def lib(tmp_path, monkeypatch):
    name = f"_runs_once_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text("LOG = []\nCONFIG = {}\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name
    sys.modules.pop(name, None)


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        ("{m}.LOG.append(v)", "{m}.LOG.append(v * 10)", ([1, 20], {})),
        ("{m}.CONFIG.update(a=v)", "{m}.LOG.extend([v, v])", ([2, 2], {"a": 1})),
    ],
    ids=["append", "update_and_extend"],
)
def test_each_change_runs_once(cash_magics, mock_shell, lib, first, second, expected):
    cells = ["import {m}", "v = 1", first, "v = 2", second, "v = 3", "z = (list({m}.LOG), dict({m}.CONFIG))"]
    cells = [cell.format(m=lib) for cell in cells]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    assert (sys.modules[lib].LOG, sys.modules[lib].CONFIG) == expected
    assert mock_shell.user_ns["z"] == expected
