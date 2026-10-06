"""An environment variable, the working directory or a module setting that a
later cell changes does not reach an earlier cell, however the later cell
changes it.

``setmode("a")`` | ``x = os.environ["MODE"] * 2`` | ``setmode("b")`` |
``print(x)``, with ``setmode`` a function of the notebook or of a local
module: the runtime watched the environment only around a statement whose
own text spelled a change (``environ``, ``chdir``), so the change
``setmode("b")`` made counted as one made outside the notebook. The upstream
check of the last cell then ran ``x`` again with ``"b"``: ``"bb"`` where a
plain kernel prints ``"aa"``. The same for ``importlib.reload(mylib)`` and
``vars(mylib)["K"] = 7``. Every statement is now watched.

A change a cell spells itself was taken as the notebook's only when the
statement's text was found in the cells, and the runtime keys the unparsed
statement: ``os.environ["MODE"] = "b"`` (as black writes it) is
``os.environ['MODE'] = 'b'`` to it. The cells are now read as the runtime
writes them (magics: ``test_a_magic_setting_below_a_reader_does_not_reach_it``).
"""

import os
import sys

import pytest

from tests._cell_driver import run_cash_cell

LIB = "import os\nK = 1\ndef set_mode(m):\n    os.environ['CASH_UT_LATER'] = m\ndef from_k(n):\n    return K * n\n"


@pytest.fixture
def lib(tmp_path, monkeypatch):
    name = f"_laterlib_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(LIB, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delenv("CASH_UT_LATER", raising=False)
    monkeypatch.chdir(tmp_path)
    yield name
    sys.modules.pop(name, None)


def _run_all(magics, cells):
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)


CASES = {
    "env_by_a_notebook_function": [
        "import os\ndef setmode(m):\n    os.environ['CASH_UT_LATER'] = m",
        "setmode('a')",
        "x = os.environ['CASH_UT_LATER'] * 2",
        "setmode('b')",
        "y = x",
    ],
    "env_by_a_module_function": [
        "import os, {lib}",
        "{lib}.set_mode('a')",
        "x = os.environ['CASH_UT_LATER'] * 2",
        "{lib}.set_mode('b')",
        "y = x",
    ],
    "cwd_by_a_notebook_function": [
        "import os\nos.makedirs('sub', exist_ok=True)\ndef go(d):\n    os.chdir(d)",
        "x = os.path.basename(os.getcwd())",
        "go('sub')",
        "y = x",
    ],
    "module_data_by_a_notebook_function": [
        "import {lib}\ndef setk(v):\n    {lib}.K = v",
        "setk(5)",
        "x = {lib}.from_k(2)",
        "setk(7)",
        "y = x",
    ],
    "module_data_by_a_reload": [
        "import importlib, {lib}",
        "{lib}.K = 5",
        "x = {lib}.from_k(2)",
        "importlib.reload({lib})",
        "y = x",
    ],
    "module_data_through_vars": [
        "import {lib}",
        "{lib}.K = 5",
        "x = {lib}.from_k(2)",
        "vars({lib})['K'] = 7",
        "y = x",
    ],
    "env_in_double_quotes": [
        "import os",
        'os.environ["CASH_UT_LATER"] = "a"',
        'x = os.environ["CASH_UT_LATER"] * 2',
        'os.environ["CASH_UT_LATER"] = "b"',
        "y = x",
    ],
    "cwd_in_double_quotes": [
        'import os\nos.makedirs("sub", exist_ok=True)',
        "x = os.path.basename(os.getcwd())",
        'os.chdir("sub")',
        "y = x",
    ],
}

EXPECTED = {
    "env_in_double_quotes": "aa",
    "cwd_in_double_quotes": None,
    "env_by_a_notebook_function": "aa",
    "env_by_a_module_function": "aa",
    "cwd_by_a_notebook_function": None,  # the folder the notebook started in
    "module_data_by_a_notebook_function": 10,
    "module_data_by_a_reload": 10,
    "module_data_through_vars": 10,
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_plain_run_all_keeps_the_earlier_value(cash_magics, mock_shell, lib, tmp_path, case):
    cells = [cell.format(lib=lib) for cell in CASES[case]]
    _run_all(cash_magics, cells)
    expected = EXPECTED[case] if EXPECTED[case] is not None else tmp_path.name
    assert mock_shell.user_ns["y"] == expected
    assert mock_shell.user_ns["x"] == expected
