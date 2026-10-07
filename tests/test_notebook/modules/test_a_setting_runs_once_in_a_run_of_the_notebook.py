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
        ("for w in [v, v]:\n    {m}.LOG.append(w)", "{m}.CONFIG['a'] = v", ([1, 1], {"a": 2})),
    ],
    ids=["append", "update_and_extend", "a_loop"],
)
def test_each_change_runs_once(cash_magics, mock_shell, lib, first, second, expected):
    cells = ["import {m}", "v = 1", first, "v = 2", second, "v = 3", "z = (list({m}.LOG), dict({m}.CONFIG))"]
    cells = [cell.format(m=lib) for cell in cells]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    assert (sys.modules[lib].LOG, sys.modules[lib].CONFIG) == expected
    assert mock_shell.user_ns["z"] == expected


@pytest.mark.parametrize("setter", ["{m}.K = 7", "{m}.CONFIG['k'] = 7", "{m}.set_k(7)"])
def test_a_setting_below_a_reader_leaves_it_alone(cash_magics, mock_shell, tmp_path, monkeypatch, setter):
    """The setting moves the module's lineage, and the simulation keyed the
    statements below it with the lineage the module ended on, not the one it
    had there: the reader above it looked changed, and ran again with 7."""
    name = f"_below_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(
        "K = 2\nCONFIG = {'k': 2}\ndef set_k(v):\n    global K\n    K = v\ndef from_k(x):\n    return x * K * CONFIG['k']\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    cells = [f"import {name}", f"b = {name}.from_k(10)", setter.format(m=name), "u = b + 1"]
    try:
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
        run_cash_cell(cash_magics, cells[-1], cells=cells)
    finally:
        sys.modules.pop(name, None)

    assert mock_shell.user_ns["u"] == 41
