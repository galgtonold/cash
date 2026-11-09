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
import types
import warnings

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


#: The logger as a module's attribute: read through the module, no call.
LOGGER = logging.getLogger(NAME)


class Holder:
    """The logger as a class's attribute, and an object's."""

    LOG = LOGGER


HOLDER = types.SimpleNamespace(log=LOGGER)

LOOKUP_SETUP = (
    "import logging\n"
    f"import {__name__} as here\n"
    f"logging.getLogger({NAME!r}).setLevel(10)\n"
    "def level_now():\n"
    f"    return logging.getLogger({NAME!r}).level\n"
    "def level_of_item(_):\n"
    "    return level_now()\n"
    "def set_level(n):\n"
    f"    logging.getLogger({NAME!r}).setLevel(n)\n"
    "    return n"
)


@pytest.mark.parametrize(
    "reader",
    [
        f"seen = logging.getLogger({NAME!r}).level",
        "seen = level_now()",
        "seen = here.LOGGER.level",
        "seen = max(map(level_of_item, [0]))",
        "seen = here.Holder.LOG.level",
        "seen = here.HOLDER.log.level",
        f"seen = logging.Logger.manager.loggerDict[{NAME!r}].level",
        f"seen = logging.Logger.manager.loggerDict.get({NAME!r}).level",
        f"seen = [lg.level for n, lg in logging.Logger.manager.loggerDict.items() if n == {NAME!r}][0]",
    ],
    ids=[
        "in-the-statement",
        "in-a-function",
        "a-module-attribute",
        "called-back-by-a-builtin",
        "a-class-attribute",
        "an-object-s-attribute",
        "the-registry-s-item",
        "the-registry-s-get",
        "the-registry-iterated",
    ],
)
def test_a_reader_looking_the_logger_up_runs_again(cash_magics, cash_instance, reader):
    """No input holds the logger: the statement, or a function it calls
    (from a builtin's code too), finds it by its name as it runs, or reads
    it through a module: an attribute of it, of a class or an object it
    holds, or an item of the registry of loggers, read by C."""
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    cells = [
        LOOKUP_SETUP,
        reader,
        "below = seen + 1",
        f"# elsewhere\nlogging.getLogger({NAME!r}).setLevel(30)",
        f"# again\n{reader}",
        "# again\nbelow = seen + 1",
    ]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["seen"] == 30
    assert cash_magics.shell.user_ns["below"] == 31


@pytest.mark.parametrize(
    "setter",
    [f"logging.getLogger({NAME!r}).setLevel(10)", "set_level(10)", "done = set_level(10)"],
    ids=["in-the-statement", "in-a-function", "in-a-kept-call"],
)
def test_a_cell_setting_the_level_the_logger_had_runs_again(cash_magics, cash_instance, setter):
    """Run when the level was already 10, it changed nothing and was stored.
    The level then set to 30 elsewhere, a hit would leave it there."""
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    cells = [LOOKUP_SETUP, setter, f"seen = logging.getLogger({NAME!r}).level"]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    run_cash_cell(cash_magics, f"# elsewhere\nlogging.getLogger({NAME!r}).setLevel(30)", cells=cells)
    for cell in cells[1:]:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert logging.getLogger(NAME).level == 10
    assert cash_magics.shell.user_ns["seen"] == 10


#: More items than cash looks through for an object found by its name
#: (`handed_callables.ITEM_TYPES_AT`), the logger last.
BIG = [0] * 100_000 + [LOGGER]
#: Loggers each found by its own name.
MANY = [logging.getLogger(f"{NAME}.{i}") for i in range(20)] + [LOGGER]


def _codes(caught):
    return [getattr(w.message, "code", None) for w in caught]


def test_a_container_too_big_to_look_through_is_never_served(cash_magics, cash_instance):
    """An item read by C from a container too big to look through: the
    statement runs every time, and a warning says why, rather than be
    served what it read before."""
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    reader = "seen = here.BIG[-1].level"
    cells = [LOOKUP_SETUP, reader, f"# elsewhere\nlogging.getLogger({NAME!r}).setLevel(30)", f"# again\n{reader}"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["seen"] == 30
    assert "NOTEBOOK-ITEMS-UNREAD" in _codes(caught)


def test_noting_past_the_budget_is_never_served(cash_magics, cash_instance, monkeypatch):
    """Checking the loggers a statement meets takes longer than its budget:
    it runs every time, and a warning says why."""
    from cash.notebook.statement.processor import StatementProcessor

    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    monkeypatch.setattr(StatementProcessor, "_noting_budget", staticmethod(lambda run: 0.0), raising=False)
    reader = "seen = here.MANY[-1].level"
    cells = [LOOKUP_SETUP, reader, f"# elsewhere\nlogging.getLogger({NAME!r}).setLevel(30)", f"# again\n{reader}"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["seen"] == 30
    assert "NOTEBOOK-ANALYSIS-BUDGET" in _codes(caught)
