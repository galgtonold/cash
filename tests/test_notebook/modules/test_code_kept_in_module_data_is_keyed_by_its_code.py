"""A function or an object of a notebook class kept in a module's data is keyed by its code.

``@mylib.register def alpha(): return 1`` in a cell, ``x = mylib.call_all()``
below it: editing ``alpha`` to return 2 and running the notebook again served
``[1]``. The module-data part of the reader's key hashed ``REG`` by value,
and a function pickles by its name, so the edited body keyed as the old one.
The same for a handler object whose class's method was edited.
"""

from __future__ import annotations

import os
import sys

import pytest

from cash.notebook.lineage_formula import module_data_digest
from cash.value_hash import compute_hash
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


def _lib(counter: str) -> str:
    return (
        "import os, time\n"
        "REG = {}\n"
        "HANDLER = None\n"
        "def _count():\n"
        f"    fd = os.open({counter!r}, os.O_WRONLY | os.O_CREAT | os.O_APPEND)\n"
        "    os.write(fd, b'x')\n"
        "    os.close(fd)\n"
        "def register(f):\n    REG[f.__name__] = f\n    return f\n"
        "def set_handler(h):\n    global HANDLER\n    HANDLER = h\n"
        f"def call_all():\n    _count()\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    return [REG[k]() for k in sorted(REG)]\n"
        f"def process(x):\n    _count()\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return HANDLER.apply(x)\n"
    )


@pytest.fixture
def lib(tmp_path, monkeypatch):
    name = f"_registry_lib_{os.getpid()}_{id(tmp_path)}"
    counter = tmp_path / "runs"
    (tmp_path / f"{name}.py").write_text(_lib(str(counter)), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name, counter
    sys.modules.pop(name, None)


CASES = {
    "decorator_registry": ("@{m}.register\ndef alpha():\n    return 1", "x = {m}.call_all()", ("return 1", "return 2")),
    "assigned_into_the_registry": (
        "def alpha():\n    return 1\n{m}.REG['alpha'] = alpha",
        "x = {m}.call_all()",
        ("return 1", "return 2"),
    ),
    "lambda_in_the_registry": ("{m}.REG['alpha'] = lambda: 1", "x = {m}.call_all()", ("lambda: 1", "lambda: 2")),
    "handler_object": (
        "class H:\n    def apply(self, x):\n        return x + 1\n{m}.set_handler(H())",
        "x = {m}.process(1)",
        ("x + 1", "x + 100"),
    ),
}

EXPECTED = {
    "decorator_registry": ([1], [2]),
    "assigned_into_the_registry": ([1], [2]),
    "lambda_in_the_registry": ([1], [2]),
    "handler_object": (2, 101),
}


def _run_all(magics, cells):
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)


@pytest.mark.parametrize("case", list(CASES))
def test_editing_the_kept_code_recomputes_the_reader(cash_magics, lib, case):
    name, _counter = lib
    setter, reader, (old, new) = CASES[case]
    cells = [f"import {name}", setter.format(m=name), reader.format(m=name)]
    _run_all(cash_magics, cells)
    assert cash_magics.shell.user_ns["x"] == EXPECTED[case][0]

    cells[1] = cells[1].replace(old, new)
    _run_all(cash_magics, cells)

    assert cash_magics.shell.user_ns["x"] == EXPECTED[case][1]


@pytest.mark.parametrize("case", ["decorator_registry", "assigned_into_the_registry"])
def test_the_reader_is_still_served_when_nothing_was_edited(cash_magics, lib, case):
    """Positive control: the code part of the key is stable, so a second
    run of the same notebook serves the reader."""
    name, counter = lib
    setter, reader, _edit = CASES[case]
    cells = [f"import {name}", setter.format(m=name), reader.format(m=name)]
    _run_all(cash_magics, cells)
    _run_all(cash_magics, cells)

    assert cash_magics.shell.user_ns["x"] == EXPECTED[case][0]
    assert len(counter.read_bytes()) == 1


def _defs(body: str) -> dict:
    # A module name nothing has loaded, as for code run in a cell.
    ns: dict = {"__name__": "_cell_of_a_test"}
    exec(compile(body, "<cell>", "exec"), ns)
    return ns


def test_a_function_s_body_moves_the_module_data_digest():
    one = _defs("def alpha():\n    return 1\n")["alpha"]
    two = _defs("def alpha():\n    return 2\n")["alpha"]

    registry = {"alpha": one}
    before = module_data_digest("m.REG", registry)
    assert module_data_digest("m.REG", registry) == before
    registry["alpha"] = two

    assert module_data_digest("m.REG", registry) != before


def test_an_object_s_class_code_moves_the_module_data_digest():
    one = _defs("class H:\n    def apply(self, x):\n        return x + 1\n")["H"]
    two = _defs("class H:\n    def apply(self, x):\n        return x + 100\n")["H"]

    holder = [one()]
    before = module_data_digest("m.HANDLER", holder)
    holder[0] = two()

    assert module_data_digest("m.HANDLER", holder) != before


@pytest.mark.parametrize(
    "value",
    [{"k": 1, "rows": [(1, 2.0, "a")] * 3}, [1, 2, 3], "text", {"nested": {"deep": [None, b"x"]}}],
    ids=["records", "list", "str", "nested"],
)
def test_plain_data_keeps_its_digest(value):
    """Data that holds no code is digested as before, so existing entries
    keep their keys."""
    assert module_data_digest("m.DATA", value) == compute_hash(value)
