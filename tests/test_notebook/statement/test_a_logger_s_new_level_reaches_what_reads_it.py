"""A statement reading a logger, or a value holding one, sees its new level.

A logger pickles by name (``getLogger(name)``), so a digest of a value
holding one holds nothing of its level, and the name finds it from anywhere.
``seen = level_of(cfg)`` -- ``cfg`` holding the logger three dicts deep --
was served 30 where plain Python gives 10 after ``log.setLevel(10)``, and
after ``logging.getLogger('app').setLevel(10)``, at the top level or in a
function, which names no variable at all. So was a statement reading
``seen`` below it. And a statement that set the level was served as a hit
when its cell ran again, leaving the level as it was.
"""

from __future__ import annotations

import logging

import pytest

from tests._cell_driver import run_cash_cell

NAME = "cash_test_logger_level"


@pytest.fixture(autouse=True)
def no_root_handlers(monkeypatch):
    # pytest's own handler on the root logger holds a stream, which no key
    # reads: the logger is then keyed without it, as in a notebook.
    monkeypatch.setattr(logging.root, "handlers", [])


def _setup(calls):
    return (
        "import logging\n"
        f"log = logging.getLogger({NAME!r})\n"
        "log.setLevel(30)\n"
        "cfg = {'a': {'b': {'c': {'log': log}}}}\n"
        f"CALLS = {str(calls)!r}\n"
        "def level_of(c):\n"
        "    with open(CALLS, 'a') as f:  # @cash:assume-safe\n"
        "        f.write('x')\n"
        "    return c['a']['b']['c']['log'].level\n"
        "def set_level(n):\n"
        f"    logging.getLogger({NAME!r}).setLevel(n)"
    )


@pytest.mark.parametrize(
    "change",
    ["log.setLevel(10)", f"logging.getLogger({NAME!r}).setLevel(10)", "set_level(10)"],
    ids=["by-variable", "by-name", "in-a-function"],
)
@pytest.mark.parametrize("reader", ["level_of(cfg)", "level_of({'a': {'b': {'c': {'log': log}}}})"])
def test_a_reader_runs_again_after_the_level_changes(cash_magics, cash_instance, tmp_path, change, reader):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    # Every call is kept as well: re-run, the reader's ``level_of(...)`` is
    # looked up in the call cache, keyed on the logger's name as the
    # statement is.
    cash_instance.config.call_cost_floor_seconds = 0.0
    calls = tmp_path / "calls"
    cells = [
        _setup(calls),
        f"seen = {reader}",
        "below = seen + 1",
        f"# served\nseen = {reader}",
        change,
        f"# again\nseen = {reader}",
        "# again\nbelow = seen + 1",
    ]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["seen"] == 10
    assert cash_magics.shell.user_ns["below"] == 11
    assert calls.read_text(encoding="utf-8") == "xx"


@pytest.mark.parametrize(
    "change",
    [f"logging.getLogger({NAME!r}).setLevel(10)", "set_level(10)", "done = set_level(10)"],
    ids=["by-name", "in-a-function", "in-a-kept-call"],
)
def test_a_statement_setting_the_level_runs_again(cash_magics, cash_instance, tmp_path, change):
    """Its cell run again after the cell above set the level back: a hit
    would leave it there, as a stored pickle of the logger holds its name.
    So would a hit of the call that set it, kept by the call cache when the
    statement binds what it returns."""
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    cells = [_setup(tmp_path / "calls"), change, "seen = level_of(cfg)"]
    for _ in range(2):
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    assert logging.getLogger(NAME).level == 10
    assert cash_magics.shell.user_ns["seen"] == 10
