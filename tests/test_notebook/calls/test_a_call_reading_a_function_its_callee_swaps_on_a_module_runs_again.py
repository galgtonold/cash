"""A call reading a function a module attribute holds is keyed on which
function that is, so a callee that swaps it is not served the first call's
answer by the second.

``m.get = one``, then ``step`` reads ``m.get`` and sets ``m.get = two``:
plain Python gives ``[1, 2]`` for ``[step(n) for n in ['a', 'a']]``. Both
calls were keyed alike -- the module keeps its lineage within the statement,
and what the key reads of a module's data leaves its functions to the code
walk, which follows their code, not which one the attribute holds -- so the
second was served the first's 1. However the attribute is set: ``setattr``,
``vars(m)[...]``, ``m.__dict__.update``, and on a module with a file.
A counter on a module (``m.n += 1``) and a function on an object were
handled already.
"""

from __future__ import annotations

import sys

import pytest

from tests._cell_driver import run_cash_cell

SETUP = "import types\nm = types.ModuleType('m')\ndef one(n):\n    return 1\ndef two(n):\n    return 2\n"

SWAPS = {
    "assigned": "{m}.get = two",
    "setattr": "setattr({m}, 'get', two)",
    "vars": "vars({m})['get'] = two",
    "dict update": "{m}.__dict__.update(get=two)",
}


@pytest.fixture
def store_everything(cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0


def _cell(module: str, swap: str) -> str:
    return (
        f"{module}.get = one\n"
        "def step(n):\n"
        f"    v = {module}.get(n)\n"
        f"    {swap.format(m=module)}\n"
        "    return v\n"
        "lv = [step(n) for n in ['a', 'a']]"
    )


def _run(cash_magics, setup: str, cell: str) -> None:
    cells = [setup, cell]
    for each in cells:
        run_cash_cell(cash_magics, each, cells=cells)


@pytest.mark.usefixtures("store_everything")
@pytest.mark.parametrize("swap", sorted(SWAPS))
def test_the_second_call_reads_the_function_the_first_left(cash_magics, mock_shell, swap):
    mock_shell.user_ns["__name__"] = "__main__"
    _run(cash_magics, SETUP, _cell("m", SWAPS[swap]))

    assert mock_shell.user_ns["lv"] == [1, 2]


@pytest.mark.usefixtures("store_everything")
def test_on_a_module_with_a_file_too(cash_magics, mock_shell, tmp_path, monkeypatch):
    mock_shell.user_ns["__name__"] = "__main__"
    (tmp_path / "swapped_get_lib.py").write_text("def get(n):\n    return 0\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "swapped_get_lib", raising=False)
    setup = SETUP + "import swapped_get_lib\n"
    _run(cash_magics, setup, _cell("swapped_get_lib", SWAPS["assigned"]))

    assert mock_shell.user_ns["lv"] == [1, 2]


@pytest.mark.usefixtures("store_everything")
def test_a_function_left_in_place_still_serves_the_second_call(cash_magics, mock_shell, tmp_path):
    """The control: nothing swapped, the second call is served."""
    mock_shell.user_ns["__name__"] = "__main__"
    log = tmp_path / "runs.log"
    cell = (
        f"m.get = one\nLOG = {str(log)!r}\n"
        "def step(n):\n"
        "    with open(LOG, 'a') as f:  # @cash:assume-safe\n"
        "        f.write('x')\n"
        "    return m.get(n)\n"
        "lv = [step(n) for n in ['a', 'a']]"
    )
    _run(cash_magics, SETUP, cell)

    assert mock_shell.user_ns["lv"] == [1, 1]
    assert log.read_text() == "x"
