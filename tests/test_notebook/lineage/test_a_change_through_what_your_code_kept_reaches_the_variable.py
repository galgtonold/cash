"""A change made through what code of the user's kept of a variable reaches
the statements that read the variable.

``keep(data["meta"])`` leaves a part of ``data`` in a list, a class
attribute, an object or a module global; a later ``bump(5)`` changes it
there, without naming ``data``. The statement reading ``data`` was keyed on
its lineage alone, which neither statement moved: once it was slow enough to
be stored (10 ms, or a loaded machine), it came back with the sum from before
the change. A stored statement now runs again and gives the new sum.
"""

from __future__ import annotations

import sys

import pytest

from tests._cell_driver import run_cash_cell

DATA = "data = {'rows': [{'v': i} for i in range(200)], 'meta': {'n': 1}}"
READ = "result = sum(r['v'] for r in data['rows']) + data['meta']['n']"

KEEPERS = """
HELD = []

def keep(d):
    HELD.append(d)

def bump(n):
    HELD[-1]['n'] += n

class Keeper:
    def __init__(self):
        self.held = []
    def keep(self, d):
        self.held.append(d)
    def bump(self, n):
        self.held[-1]['n'] += n

class Registry:
    held = []
    @staticmethod
    def keep(d):
        Registry.held.append(d)
    @staticmethod
    def bump(n):
        Registry.held[-1]['n'] += n
    @staticmethod
    def keep_in_module(d):
        HELD.append(d)
    @staticmethod
    def bump_in_module(n):
        HELD[-1]['n'] += n
"""

IN_CELLS = "HELD = []\ndef keep(d):\n    HELD.append(d)\ndef bump(n):\n    HELD[-1]['n'] += n"

CASES = {
    "a global of the notebook a function keeps it in": ([IN_CELLS], ["keep(data['meta'])", "bump(5)"]),
    "an object of the user's class": (
        ["from keepers import Keeper", "kp = Keeper()"],
        ["kp.keep(data['meta'])", "kp.bump(5)"],
    ),
    "a class attribute": (["from keepers import Registry"], ["Registry.keep(data['meta'])", "Registry.bump(5)"]),
    "a module global, through the module": (["import keepers"], ["keepers.keep(data['meta'])", "keepers.bump(5)"]),
    "a module global, through a class of it": (
        ["from keepers import Registry"],
        ["Registry.keep_in_module(data['meta'])", "Registry.bump_in_module(5)"],
    ),
}


@pytest.fixture
def notebook(cash_magics, cash_instance, tmp_path, monkeypatch):
    """Every statement stored, however fast, and a module of the user's."""
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    (tmp_path / "keepers.py").write_text(KEEPERS, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.delitem(sys.modules, "keepers", raising=False)
    cash_magics.cash_on("")
    cash_magics.badges.mode = "off"
    yield cash_magics
    sys.modules.pop("keepers", None)


def _expected(ns: dict) -> int:
    return sum(r["v"] for r in ns["data"]["rows"]) + ns["data"]["meta"]["n"]


@pytest.mark.parametrize("how", list(CASES))
def test_the_reading_statement_gives_the_changed_value(notebook, how):
    setup, change = CASES[how]
    ns = notebook.shell.user_ns
    before_change = [*setup, DATA, READ]
    cells = [*setup, DATA, *change, READ]
    for cell in [*setup, DATA]:
        run_cash_cell(notebook, cell, cells=before_change)
    for _ in range(2):  # stored, then served
        run_cash_cell(notebook, READ, cells=before_change)
    before = ns["result"]
    for cell in change:
        run_cash_cell(notebook, cell, cells=cells)
    run_cash_cell(notebook, READ, cells=cells)
    assert ns["result"] == _expected(ns) == before + 5


def test_code_that_kept_nothing_leaves_the_reading_statement_stored(notebook, monkeypatch):
    """Control: the link is made only where the code kept a part of the
    variable, so a change to what it keeps otherwise moves nothing."""
    setup = ["from keepers import Registry", "other = {'n': 0}"]
    change = ["Registry.keep(other)", "Registry.bump(5)"]
    before_change = [*setup, DATA, READ]
    cells = [*setup, DATA, *change, READ]
    for cell in [*setup, DATA]:
        run_cash_cell(notebook, cell, cells=before_change)
    for _ in range(2):
        run_cash_cell(notebook, READ, cells=before_change)
    for cell in change:
        run_cash_cell(notebook, cell, cells=cells)
    analytics = notebook._statement_processor.analytics_manager
    statuses: list[str] = []
    record = analytics.record_event
    monkeypatch.setattr(
        analytics, "record_event", lambda status, *a, **k: (statuses.append(status), record(status, *a, **k))
    )
    run_cash_cell(notebook, READ, cells=cells)
    assert statuses == ["HIT"], f"the reading statement ran again, though nothing it reads changed: {statuses}"
