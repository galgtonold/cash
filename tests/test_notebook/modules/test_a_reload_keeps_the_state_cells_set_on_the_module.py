"""Reloading an edited local module keeps the state the notebook's cells set on it.

``helper.SCALE = 5`` in one cell, ``helper.compute(3)`` in the next: editing
an unrelated function in ``helper.py`` reloaded the module, which ran its top
level again and put ``SCALE`` back to 1. The cell that set it was not run
again, so the next cell computed with the file's default, a state neither a
plain kernel nor a top-to-bottom run has. The statements that set state on
the module now run again right after the reload.
"""

import os
import sys
import warnings

import pytest

from cash.exceptions import CashWarning, UpstreamStateError
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

HELPER = (
    "import time\n"
    "SCALE = 1\n"
    "REGISTRY = {{}}\n"
    "def set_scale(v):\n    global SCALE\n    SCALE = v\n"
    f"def compute(v):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return (v * SCALE, dict(REGISTRY))\n"
    "def unrelated():\n    return {n}\n"
)


@pytest.fixture
def helper_module(tmp_path, monkeypatch):
    name = f"_reload_state_{os.getpid()}_{id(tmp_path)}"
    path = tmp_path / f"{name}.py"
    path.write_text(HELPER.format(n=1), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield name, path
    sys.modules.pop(name, None)


def _edit(path, text):
    before = os.stat(path).st_mtime_ns
    path.write_text(text, encoding="utf-8")
    # A reload is decided by the file's mtime, which must move.
    os.utime(path, ns=(before + 2_000_000_000, before + 2_000_000_000))


@pytest.mark.parametrize(
    "setter",
    [
        "{m}.SCALE = 5\n{m}.REGISTRY['a'] = 10",
        "setattr({m}, 'SCALE', 5)\n{m}.REGISTRY.update(a=10)",
        "{m}.set_scale(5)\n{m}.REGISTRY['a'] = 10",
    ],
    ids=["assignment", "setattr_and_method", "module_function"],
)
def test_editing_an_unrelated_function_keeps_the_settings(cash_magics, mock_shell, helper_module, setter):
    name, path = helper_module
    cells = [f"import {name}", setter.format(m=name), f"z = {name}.compute(3)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert mock_shell.user_ns["z"] == (15, {"a": 10})

    _edit(path, HELPER.format(n=2))
    run_cash_cell(cash_magics, cells[2], cells=cells)

    assert mock_shell.user_ns["z"] == (15, {"a": 10})


def test_a_setting_that_no_longer_runs_is_reported(cash_magics, mock_shell, helper_module):
    name, path = helper_module
    cells = [f"import {name}", f"{name}.set_scale(5)", f"z = {name}.compute(3)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    _edit(path, HELPER.format(n=1).replace("global SCALE\n    SCALE = v", "raise ValueError('no longer settable')"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # The upstream check runs the setting again too, and stops the cell.
        with pytest.raises(UpstreamStateError):
            run_cash_cell(cash_magics, cells[2], cells=cells)

    codes = [getattr(w.message, "code", None) for w in caught if isinstance(w.message, CashWarning)]
    assert "NOTEBOOK-RELOAD-STATE" in codes


@pytest.mark.parametrize(
    "cells",
    [
        ["import {m}", "k = 5", "{m}.SCALE = k", "k = 9"],
        ["import {m}, random\nrng = random.Random(0)", "{m}.SCALE = rng.randrange(100)", "a = 1"],
    ],
    ids=["an_input_rebound_since", "a_draw_in_the_setting"],
)
def test_a_setting_whose_inputs_moved_on_keeps_the_value_it_set(cash_magics, mock_shell, helper_module, cells):
    """Run again, ``helper.SCALE = k`` would read the ``k`` a later cell bound,
    and a draw would draw anew: neither is the value the notebook set."""
    name, path = helper_module
    cells = [cell.format(m=name) for cell in cells] + [f"z = {name}.compute(3)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    module = sys.modules[name]
    scale = module.SCALE
    rng_state = mock_shell.user_ns["rng"].getstate() if "rng" in mock_shell.user_ns else None

    _edit(path, HELPER.format(n=2))
    run_cash_cell(cash_magics, cells[-1], cells=cells)

    assert sys.modules[name].SCALE == scale
    assert mock_shell.user_ns["z"] == (3 * scale, {})
    if rng_state is not None:
        assert mock_shell.user_ns["rng"].getstate() == rng_state


def test_a_setting_that_reads_a_rebound_input_and_binds_no_attribute_is_reported(
    cash_magics, mock_shell, helper_module
):
    name, path = helper_module
    cells = [f"import {name}", "k = 5", f"{name}.set_scale(k)", "k = 9", f"z = {name}.compute(3)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)

    _edit(path, HELPER.format(n=2))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, cells[-1], cells=cells)

    codes = [getattr(w.message, "code", None) for w in caught if isinstance(w.message, CashWarning)]
    assert "NOTEBOOK-RELOAD-STATE" in codes
    assert sys.modules[name].SCALE != 9
