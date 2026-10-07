"""Reloading an edited local module keeps the state the notebook's cells set on it.

``helper.SCALE = 5`` in one cell, ``helper.compute(3)`` in the next: editing
an unrelated function in ``helper.py`` reloaded the module, which ran its top
level again and put ``SCALE`` back to 1. The cell that set it was not run
again, so the next cell computed with the file's default, a state neither a
plain kernel nor a top-to-bottom run has.

Running the setting again in place was not the answer either: it read what
its inputs hold now, so ``helper.SCALE = k`` with ``k`` rebound by a later
cell set 9 where the notebook set 5, and a draw drew anew. The module is now
a value of the notebook like any other: ``import helper`` makes it and each
statement that sets state on it changes it in place, so a reload leaves it
where the import alone does, and the upstream check rebuilds it from those
statements as it rebuilds a variable -- each with the inputs it had, a
generator put back where it drew from it.
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
    "LOG = []\n"
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


def _run_all(cash_magics, cells):
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)


def _codes(caught):
    return [getattr(w.message, "code", None) for w in caught if isinstance(w.message, CashWarning)]


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
    _run_all(cash_magics, cells)
    assert mock_shell.user_ns["z"] == (15, {"a": 10})

    _edit(path, HELPER.format(n=2))
    run_cash_cell(cash_magics, cells[2], cells=cells)

    assert mock_shell.user_ns["z"] == (15, {"a": 10})


def test_a_setting_that_no_longer_runs_stops_the_cell(cash_magics, mock_shell, helper_module):
    """Rebuilding the module runs the setting again; the edit made it raise,
    so the cell stops, as it does for any statement a rebuild runs."""
    name, path = helper_module
    cells = [f"import {name}", f"{name}.set_scale(5)", f"z = {name}.compute(3)"]
    _run_all(cash_magics, cells)

    _edit(path, HELPER.format(n=1).replace("global SCALE\n    SCALE = v", "raise ValueError('no longer settable')"))
    with pytest.raises(UpstreamStateError, match="no longer settable"):
        run_cash_cell(cash_magics, cells[2], cells=cells)


@pytest.mark.parametrize(
    ("cells", "scale", "registry"),
    [
        (["import {m}", "k = 5", "{m}.SCALE = k", "k = 9"], 5, {}),
        (["import {m}", "k = 5", "{m}.set_scale(k)", "k = 9"], 5, {}),
        (["import {m}", "v = 7", "{m}.REGISTRY['a'] = v", "v = 8"], 1, {"a": 7}),
        (["import {m}", "k = 5", "{m}.set_scale(k)", "k = {m}.SCALE + 1", "{m}.SCALE = k * 2", "k = 0"], 12, {}),
    ],
    ids=["an_attribute_from_a_rebound_input", "a_module_function_on_a_rebound_input", "an_item", "chained"],
)
def test_a_setting_is_rebuilt_with_the_inputs_it_had(cash_magics, mock_shell, helper_module, cells, scale, registry):
    """Run again in place, the setting read the ``k`` a later cell bound.
    Rebuilt, it reads the ``k`` it read when it ran, as a top-to-bottom run
    does, and a setting that reads the module's own state reads it as the
    settings above it left it."""
    name, path = helper_module
    cells = [cell.format(m=name) for cell in cells] + [f"z = {name}.compute(3)"]
    _run_all(cash_magics, cells)
    assert mock_shell.user_ns["z"] == (3 * scale, registry)
    k = mock_shell.user_ns.get("k")

    _edit(path, HELPER.format(n=2))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, cells[-1], cells=cells)

    assert "NOTEBOOK-RELOAD-STATE" not in _codes(caught)
    assert sys.modules[name].SCALE == scale
    assert mock_shell.user_ns["z"] == (3 * scale, registry)
    assert mock_shell.user_ns.get("k") == k


@pytest.mark.parametrize("setter", ["{m}.SCALE = rng.randrange(100)", "{m}.set_scale(rng.randrange(100))"])
def test_a_drawn_setting_is_rebuilt_from_where_it_drew(cash_magics, mock_shell, helper_module, setter):
    """The draw is made again from the generator state it drew from, so the
    module gets the same number, and the generator ends where it was."""
    name, path = helper_module
    cells = [f"import {name}, random\nrng = random.Random(0)", setter.format(m=name), "a = 1", f"z = {name}.compute(3)"]
    _run_all(cash_magics, cells)
    scale = sys.modules[name].SCALE
    rng_state = mock_shell.user_ns["rng"].getstate()

    _edit(path, HELPER.format(n=2))
    run_cash_cell(cash_magics, cells[-1], cells=cells)

    assert sys.modules[name].SCALE == scale
    assert mock_shell.user_ns["z"] == (3 * scale, {})
    assert mock_shell.user_ns["rng"].getstate() == rng_state


def test_an_object_of_the_module_changed_in_place_is_rebuilt_once(cash_magics, mock_shell, helper_module):
    """``helper.LOG.append(v)`` with ``v`` rebound since: the list is
    rebuilt with the value appended then, and running the reader again does
    not append it twice."""
    name, path = helper_module
    cells = [f"import {name}", "v = 1", f"{name}.LOG.append(v)", "v = 2", f"{name}.LOG.append(v * 10)", "v = 3"]
    cells.append(f"z = {name}.compute(3)")
    _run_all(cash_magics, cells)
    assert sys.modules[name].LOG == [1, 20]

    _edit(path, HELPER.format(n=2))
    run_cash_cell(cash_magics, cells[-1], cells=cells)
    run_cash_cell(cash_magics, cells[-1], cells=cells)

    assert sys.modules[name].LOG == [1, 20]
    assert mock_shell.user_ns["v"] == 3


@pytest.mark.parametrize(
    "setter",
    [
        "for v in vals:\n    {m}.LOG.append(v)",
        "if vals:\n    {m}.SCALE = vals[0]",
        "def set_first(v):\n    {m}.SCALE = v\nset_first(vals[0])",
    ],
    ids=["a_loop", "a_branch", "a_notebook_function"],
)
def test_a_setting_in_a_loop_a_branch_or_a_function_is_rebuilt(cash_magics, mock_shell, helper_module, setter):
    """The loop or branch changes the module as it changes a list it fills;
    a function of the notebook's that sets it changes it when called, not
    when defined."""
    name, path = helper_module
    cells = [f"import {name}", "vals = [4, 2]", setter.format(m=name), "vals = [9]", f"z = {name}.compute(3)"]
    _run_all(cash_magics, cells)
    module = sys.modules[name]
    state = (list(module.LOG), module.SCALE)
    assert state != ([], 1)

    _edit(path, HELPER.format(n=2))
    run_cash_cell(cash_magics, cells[-1], cells=cells)

    module = sys.modules[name]
    assert (module.LOG, module.SCALE) == state


def test_the_state_is_rebuilt_whatever_the_cell_reads(cash_magics, mock_shell, helper_module):
    name, path = helper_module
    cells = [f"import {name}", "k = 5", f"{name}.SCALE = k", "k = 9", "y = 2"]
    _run_all(cash_magics, cells)

    _edit(path, HELPER.format(n=2))
    run_cash_cell(cash_magics, cells[-1], cells=cells)

    assert sys.modules[name].SCALE == 5


def test_a_setting_imported_from_the_module_is_rebuilt(cash_magics, mock_shell, helper_module):
    """``set_scale(k)`` imported from the module changes the module the
    notebook holds as ``helper``, though it does not name it."""
    name, path = helper_module
    cells = [f"import {name}\nfrom {name} import set_scale", "k = 5", "set_scale(k)", "k = 9", f"z = {name}.compute(3)"]
    _run_all(cash_magics, cells)

    _edit(path, HELPER.format(n=2))
    run_cash_cell(cash_magics, cells[-1], cells=cells)

    assert sys.modules[name].SCALE == 5
    assert mock_shell.user_ns["z"] == (15, {})


def test_a_module_held_only_through_what_was_imported_from_it_is_rebuilt(cash_magics, mock_shell, helper_module):
    """Only ``from helper import ...``: no name of the notebook is the
    module, but ``compute`` reads its state and ``set_scale`` changes it, so
    they are the names the state is rebuilt through."""
    name, path = helper_module
    cells = [f"from {name} import set_scale, compute", "k = 5", "set_scale(k)", "k = 9", "z = compute(3)"]
    _run_all(cash_magics, cells)

    _edit(path, HELPER.format(n=2))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, cells[-1], cells=cells)

    assert "NOTEBOOK-RELOAD-STATE" not in _codes(caught)
    assert sys.modules[name].SCALE == 5
    assert mock_shell.user_ns["z"] == (15, {})


def test_a_setting_only_running_it_shows_is_rebuilt(cash_magics, mock_shell, helper_module):
    """``setattr(sys.modules[__name__], ...)`` in a module function: no text
    says it sets state, but running it was seen rebinding the module's
    global, so it is a setting like the others."""
    name, path = helper_module
    text = HELPER + "def put(attr, v):\n    import sys\n    setattr(sys.modules[__name__], attr, v)\n"
    path.write_text(text.format(n=1), encoding="utf-8")
    cells = [f"import {name}", f"{name}.put('SCALE', 5)", f"z = {name}.compute(3)"]
    _run_all(cash_magics, cells)
    assert mock_shell.user_ns["z"] == (15, {})

    _edit(path, text.format(n=2))
    run_cash_cell(cash_magics, cells[-1], cells=cells)

    assert sys.modules[name].SCALE == 5
    assert mock_shell.user_ns["z"] == (15, {})


def test_a_module_no_name_sees_is_reported(cash_magics, mock_shell, helper_module, tmp_path):
    """Set through another module: no name of the notebook sees the module,
    so nothing rebuilds what was set on it, and cash says so."""
    name, path = helper_module
    outer = f"{name}_outer"
    (tmp_path / f"{outer}.py").write_text(f"import {name}\n", encoding="utf-8")
    cells = [f"import {outer}", f"{outer}.{name}.set_scale(5)", f"z = {outer}.{name}.compute(3)"]
    try:
        _run_all(cash_magics, cells)

        _edit(path, HELPER.format(n=2))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            run_cash_cell(cash_magics, cells[-1], cells=cells)
    finally:
        sys.modules.pop(outer, None)

    assert "NOTEBOOK-RELOAD-STATE" in _codes(caught)


def test_without_the_notebook_the_lost_state_is_reported(cash_magics, mock_shell, helper_module):
    """With no notebook to simulate, nothing rebuilds the module: cash says
    so rather than compute on the file's values unannounced."""
    name, path = helper_module
    cells = [f"import {name}", f"{name}.SCALE = 5", f"z = {name}.compute(3)"]
    for cell in cells:
        run_cash_cell(cash_magics, cell)

    _edit(path, HELPER.format(n=2))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, cells[-1])

    assert "NOTEBOOK-RELOAD-STATE" in _codes(caught)
