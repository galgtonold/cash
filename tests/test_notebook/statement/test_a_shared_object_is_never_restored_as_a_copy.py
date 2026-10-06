"""A statement whose outputs hold an object something else holds too is never
restored as a copy.

A hit binds each output name to a deserialised copy. When another variable or
container also holds the object (or an object inside it), that holder keeps the
object the statement really produced or changed, and the change is lost. So
the holder is stored and restored with the outputs when it is a variable of
the notebook (test_an_alias_between_variables_is_restored_with_it), and the
statement runs again when it is anything else, or in a loop body:

* ``d = dfs[0]`` then ``d['a'] = slow(...)``: the hit rebinds ``d`` to a changed
  copy, and ``dfs[0]`` keeps its old contents on the next Run All;
* ``for b in bufs: b += slow(...)`` over equal buffers: the second iteration
  hits the first one's entry and only rebinds ``b``, already on the first run;
* ``models = {'m': m, ...}``: restored, the dict holds a copy of ``m``, and a
  later ``m.fit()`` is invisible through it.

Each case runs "Run All" twice and must match plain Python both times.
"""

from __future__ import annotations

import types

import pytest

from cash.notebook.cache_status import CacheStatus
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

SETUP = f"import time\ndef slow(x):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n    return x"
MODEL = (
    f"{SETUP}\nclass Model:\n    def __init__(self):\n        self.w = 0\n"
    f"    def fit(self):\n        time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n        self.w += 1"
)

NOTEBOOKS = {
    "an item taken out of a list": (
        [SETUP, "rows = [{'a': 1.0}, {'a': 3.0}]", "d = rows[0]\nd['a'] = slow(d['a'] * 2)", "r = rows[0]['a']"],
        2.0,
    ),
    "a list unpacked": (
        [SETUP, "rows = [{'a': 1.0}, {'a': 3.0}]", "x, y = rows\nx['a'] = slow(x['a'] * 2)", "r = rows[0]['a']"],
        2.0,
    ),
    "a value of a dict": (
        [
            SETUP,
            "cfg = {'model': {'lr': 1}}",
            "m = cfg['model']\nm['score'] = slow(m['lr'] * 10)",
            "r = cfg['model'].get('score')",
        ],
        10,
    ),
    "equal buffers changed in a loop": (
        [
            SETUP,
            "bufs = [[0.0] for _ in range(3)]",
            "for b in bufs:\n    b += slow([1.0])",
            "r = sum(len(b) for b in bufs)",
        ],
        6,
    ),
    "a model put in a dict": (
        [MODEL, "m = Model()", "models = {'m': m, 'tag': slow('v1')}", "m.fit()", "r = models['m'].w"],
        1,
    ),
    "a list put in a list": (
        [SETUP, "data = [1, 2]", "reg = [data, slow('meta')]", "data.append(3)", "r = len(reg[0])"],
        3,
    ),
}


def _run_all(magics, cells):
    _run_all_cells(magics, cells)
    return magics.shell.user_ns["r"]


@pytest.mark.parametrize("name", list(NOTEBOOKS))
def test_a_second_run_all_keeps_the_change(cash_magics, name):
    cells, expected = NOTEBOOKS[name]
    assert _run_all(cash_magics, cells) == expected, "first Run All"
    assert _run_all(cash_magics, cells) == expected, "second Run All"


def _stored(metrics) -> bool:
    return metrics["status"] == CacheStatus.COMPUTED and not metrics.get("uncacheable_reasons")


def test_an_object_made_in_the_same_statement_is_stored(cash_magics, statement_processor):
    """Control: nothing else holds the dicts, so the statement is cached."""
    run_cash_cell(cash_magics, MODEL)
    metrics = statement_processor.process_statement("models = {'m': Model(), 'tag': slow('v1')}")
    assert _stored(metrics), metrics.get("uncacheable_reasons")
    metrics = statement_processor.process_statement("models['tag'] = slow('v2')")
    assert _stored(metrics), metrics.get("uncacheable_reasons")


def test_a_shared_output_says_why_it_runs_again(cash_magics, statement_processor):
    """A closure holds the model: no restore can rebind it. (A notebook
    variable holding it is stored with the outputs instead, see
    test_an_alias_between_variables_is_restored_with_it.)"""
    run_cash_cell(cash_magics, MODEL)
    run_cash_cell(cash_magics, "m = Model()\nkeep = (lambda held: lambda: held)(m)")
    metrics = statement_processor.process_statement("models = {'m': m, 'tag': slow('v1')}")
    reasons = " ".join(metrics.get("uncacheable_reasons") or [])
    assert "'models' holds an object another variable or container also holds" in reasons, reasons


def test_a_library_module_holding_the_object_still_refuses(cash_magics, statement_processor):
    """A module of a library holds the dict: no restore can rebind it."""
    run_cash_cell(cash_magics, f"{SETUP}\nimport json")
    statement_processor.process_statement("a = {'v': 1}")
    statement_processor.process_statement("json._held_by_a_test = a")
    try:
        metrics = statement_processor.process_statement(f"a['x'] = (time.sleep({ABOVE_PERSISTENCE_FLOOR_S}), 1)[1]")
    finally:
        statement_processor.process_statement("del json._held_by_a_test")
    reasons = " ".join(metrics["uncacheable_reasons"])
    assert "'a' holds an object another variable or container also holds" in reasons, reasons


def _run_all_cells(magics, cells):
    for cell in cells:
        run_cash_cell(magics, cell, cells=cells)


def _show(user_ns, n, value, *, ipython=True):
    """What IPython's displayhook leaves behind after showing *value* as Out[n]."""
    out = {n: value}
    user_ns["Out"] = out
    user_ns["_oh"] = out if ipython else {}
    user_ns["_"] = user_ns[f"_{n}"] = value


def _after_showing(cash_magics, statement_processor, **show):
    run_cash_cell(cash_magics, SETUP)
    run_cash_cell(cash_magics, "rows = [1.0]")
    _show(statement_processor.shell.user_ns, 2, statement_processor.shell.user_ns["rows"], **show)
    return statement_processor.process_statement("rows += slow([2.0])")


def test_the_output_history_is_not_a_holder(cash_magics, statement_processor):
    """``rows`` was displayed, so ``Out[2]``, ``_`` and ``_2`` hold it too.
    Nothing in a program depends on the history holding the very object, so
    the statement is still stored instead of running every time."""
    metrics = _after_showing(cash_magics, statement_processor)
    assert _stored(metrics), metrics.get("uncacheable_reasons")


def test_a_dict_named_out_that_is_not_the_history_is_a_holder(cash_magics, statement_processor):
    metrics = _after_showing(cash_magics, statement_processor, ipython=False)
    assert not _stored(metrics)


def test_the_shells_copies_of_the_output_history_are_not_holders(cash_magics, statement_processor):
    """IPython's display hook binds a shown value three more times outside
    ``user_ns``: its own ``_`` attribute, ``user_ns_hidden`` (``push`` with
    ``interactive=False``), and the cell's ``last_execution_result``. Once a
    cell's result went through the hook, each of them made the statement
    updating ``rows`` in the next cell run every time."""
    run_cash_cell(cash_magics, SETUP)
    run_cash_cell(cash_magics, "rows = [1.0]")
    shell = statement_processor.shell
    rows = shell.user_ns["rows"]
    _show(shell.user_ns, 2, rows)
    shell.displayhook = types.SimpleNamespace(_=rows, __=None, ___=None)
    shell.user_ns_hidden = {"_": rows, "_2": rows}
    shell.last_execution_result = types.SimpleNamespace(result=rows)
    del rows

    metrics = statement_processor.process_statement("rows += slow([2.0])")

    assert _stored(metrics), metrics.get("uncacheable_reasons")


@pytest.mark.parametrize("shown", ["frames = [d1, d2]", "reg = {'cfg': d1}"])
def test_a_container_that_was_shown_still_holds_the_object(cash_magics, statement_processor, shown):
    """A cell ending in ``frames`` puts the list in ``Out`` too. ``Out`` is
    not a holder, but ``frames`` is a variable all the same: its reference
    to ``d1`` is a holder's. Counted as the history's, the update of ``d1``
    was restored on the next Run All as a copy, and ``frames[0]`` kept the
    old object without ``z``."""
    name = shown.split()[0]
    cells = [SETUP, "d1 = {'a': 1}\nd2 = {'a': 3}", shown, "d1['z'] = slow(d1['a'] * 2)"]
    user_ns = statement_processor.shell.user_ns
    for run in ("first", "second"):
        for i, cell in enumerate(cells):
            run_cash_cell(cash_magics, cell, cells=cells)
            if i == 2:
                _show(user_ns, 3, user_ns[name])
        held = user_ns[name][0 if name == "frames" else "cfg"]
        assert held is user_ns["d1"], f"{run} Run All"
        assert held.get("z") == 2, f"{run} Run All"


TRACKER = f"{SETUP}\nclass Tracker:\n    def __init__(self):\n        self.items = []\n    def log(self, v):\n        self.items.append(v)"
HOOKS = {
    "a bound method": (["tracker = Tracker()", "hooks = {'log': tracker.log, 'n': slow(1)}", "hooks['log'](5)"], "tracker.items"),
    "a builtin bound method": (["results = []", "hooks = {'add': results.append, 'n': slow(1)}", "hooks['add'](5)"], "results"),
    "a closure": (
        ["store = []\ndef make(lst):\n    return lambda v: lst.append(v)", "hooks = {'f': make(store), 'n': slow(1)}", "hooks['f'](5)"],
        "store",
    ),
}


@pytest.mark.parametrize("name", list(HOOKS))
def test_a_hook_writes_into_the_notebooks_object(cash_magics, name):
    """A bound method pickles its object by value, and a deep copy keeps a
    closure, or a builtin method like ``results.append``, bound to the
    object of the run that stored it. Restored, the hook wrote into a copy
    or into the last run's object, and ``tracker.items`` stayed empty."""
    cells, probe = HOOKS[name]
    cells = [TRACKER, *cells, f"r = list({probe})"]
    for run in ("first", "second"):
        _run_all_cells(cash_magics, cells)
        assert eval(probe, cash_magics.shell.user_ns) == [5], f"{run} Run All"
