"""A logger a module holds deep in a container, or a library hands back in one.

``seen = mod.CFG["a"]["log"].level`` reads the logger two items down a dict
a module holds, and ``seen = mod.ITEMS[0].log.level`` an attribute of an
object a list holds; ``seen = lib.settings()["log"].level`` reads it inside
what a library's function returns. No input holds it, and only the first
level of a container read through a module was looked through, and only a
library's return that was the logger itself: each was served the level it
read before (10 where plain Python gives 30) once another cell changed it.
"""

from __future__ import annotations

import collections
import logging
import os
import sys
import types
import warnings

import pytest

from tests._cell_driver import run_cash_cell

needs_monitoring = pytest.mark.skipif(
    sys.version_info < (3, 12),
    reason="sys.monitoring needs Python 3.12: below it what a statement finds by a lookup is not watched "
    "(NOTEBOOK-LOOKUPS-UNSEEN)",
)

NAME = "cash_test_logger_deep"
LOGGER = logging.getLogger(NAME)


@pytest.fixture(autouse=True)
def no_root_handlers(monkeypatch):
    # pytest's own handler on the root logger holds a stream, which no key
    # reads: the logger is then keyed without it, as in a notebook.
    monkeypatch.setattr(logging.root, "handlers", [])


class Plain:
    """An object of the user's holding the logger as an attribute."""

    def __init__(self):
        self.log = LOGGER


Pair = collections.namedtuple("Pair", "name log")


class Subclassed(dict):
    pass


#: Held by this module, read through it.
CFG = {"a": {"log": LOGGER, "n": 1}}
ITEMS = [Plain()]
DEEP = [[[{"x": (0, LOGGER)}]]]
PAIRS = [Pair("app", LOGGER)]
KEYED = {"by": {LOGGER: 1}}
SUBCLASSED = {"a": Subclassed(log=LOGGER)}
#: Only numbers and strings, more items than are looked through one by one.
ROWS = [{"id": i, "v": [i, i, i]} for i in range(60_000)]
#: More objects than are looked through, the logger last.
WIDE = {"rows": [types.SimpleNamespace(i=i) for i in range(60_000)] + [types.SimpleNamespace(log=LOGGER)]}

#: Functions of a library's: their code is in a file of the installation.
_LIBRARY = os.path.join(os.path.dirname(logging.__file__), "_cash_test_settings.py")
_LIB_SOURCE = """
import dataclasses, logging, types

@dataclasses.dataclass
class Record:
    log: object

def _found():
    # By its name, as a library finds its logger: no global of its holds it.
    return logging.getLogger(NAME)

def settings():
    return {"log": _found(), "n": 1}

def nested():
    return {"a": [types.SimpleNamespace(log=_found())]}

def holder():
    return types.SimpleNamespace(log=_found())

def record():
    return Record(_found())

def in_a_tuple():
    return ("app", [_found()])

def too_much():
    return [types.SimpleNamespace(i=i) for i in range(60_000)] + [types.SimpleNamespace(log=_found())]
"""
#: A library's module: its file is in the installation.
lib = types.ModuleType("cash_test_library_settings")
lib.__file__ = _LIBRARY
lib.NAME = NAME
sys.modules[lib.__name__] = lib
exec(compile(_LIB_SOURCE, _LIBRARY, "exec"), vars(lib))

SETUP = f"import logging\nimport {__name__} as here\nhere.LOGGER.setLevel(10)"


def _run(cash_magics, cash_instance, reader):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    cells = [
        SETUP,
        reader,
        "below = seen + 1",
        "# elsewhere\nhere.LOGGER.setLevel(30)",
        f"# again\n{reader}",
        "# again\nbelow = seen + 1",
    ]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for cell in cells:
            run_cash_cell(cash_magics, cell, cells=cells)
    return [getattr(w.message, "code", None) for w in caught]


@needs_monitoring
@pytest.mark.parametrize(
    "reader",
    [
        "seen = here.CFG['a']['log'].level",
        "seen = here.ITEMS[0].log.level",
        "seen = here.DEEP[0][0][0]['x'][1].level",
        "seen = here.PAIRS[0].log.level",
        "seen = [lg for lg in here.KEYED['by']][0].level",
        "seen = here.SUBCLASSED['a']['log'].level",
    ],
    ids=[
        "two-items-down",
        "an-item-s-attribute",
        "five-down",
        "a-named-tuple-s-field",
        "a-dict-s-key",
        "a-dict-subclass-s-item",
    ],
)
def test_a_logger_deep_in_what_a_module_holds_is_checked(cash_magics, cash_instance, reader):
    """Read through a module more than one item down: the statement, and
    the one reading what it bound, run again once the level changed."""
    _run(cash_magics, cash_instance, reader)
    assert cash_magics.shell.user_ns["seen"] == 30
    assert cash_magics.shell.user_ns["below"] == 31


@pytest.fixture
def fresh_kernel(monkeypatch):
    """What a kernel that just started knows: no class seen pickled by its
    name, no library function (``logging.getLogger``) seen returning one,
    so only what the library's function returns can show the logger."""
    from cash import _weak, kept_state
    from cash.tracking import lookups_seen

    monkeypatch.setattr(kept_state, "_KINDS_BY_REFERENCE", set())
    monkeypatch.setattr(lookups_seen, "_RETURNERS", _weak.WeakSet())


@needs_monitoring
@pytest.mark.usefixtures("fresh_kernel")
@pytest.mark.parametrize(
    "reader",
    [
        "seen = here.lib.settings()['log'].level",
        "seen = here.lib.nested()['a'][0].log.level",
        "seen = here.lib.holder().log.level",
        "seen = here.lib.record().log.level",
        "seen = here.lib.in_a_tuple()[1][0].level",
    ],
    ids=["a-dict", "a-dict-of-lists", "a-namespace", "a-dataclass", "a-tuple"],
)
def test_a_logger_in_what_a_library_returns_is_checked(cash_magics, cash_instance, reader):
    """A library's function hands the logger back inside a container or a
    plain object: the statement runs again once the level changed."""
    _run(cash_magics, cash_instance, reader)
    assert cash_magics.shell.user_ns["seen"] == 30
    assert cash_magics.shell.user_ns["below"] == 31


@needs_monitoring
@pytest.mark.usefixtures("fresh_kernel")
@pytest.mark.parametrize(
    "reader",
    ["seen = here.WIDE['rows'][-1].log.level", "seen = here.lib.too_much()[-1].log.level"],
    ids=["held-by-a-module", "returned-by-a-library"],
)
def test_more_than_is_looked_through_runs_every_time_and_says_so(cash_magics, cash_instance, reader):
    """Past the items looked through, the statement is never served: it
    runs every time, and a warning says why."""
    codes = _run(cash_magics, cash_instance, reader)
    assert cash_magics.shell.user_ns["seen"] == 30
    assert "NOTEBOOK-ITEMS-UNREAD" in codes


@needs_monitoring
@pytest.mark.parametrize(
    "reader",
    [
        "seen = here.CFG['a']['n']",
        "seen = here.ROWS[5]['v'][0]",
        "seen = here.lib.settings()['n']",
        "seen = here.CFG['a']['log'].level",
    ],
    ids=["a-number-beside-it", "rows-of-numbers", "a-library-s-dict", "the-logger"],
)
def test_a_read_is_served_while_nothing_changed(cash_magics, cash_instance, reader):
    """Looked through deeper, a read is still served on the next run all,
    and many rows of numbers are read at once, not refused."""
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cells = [SETUP, reader]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        for _ in range(2):
            for cell in cells:
                run_cash_cell(cash_magics, cell, cells=cells)
    statuses = [str(s["status"]) for s in cash_magics._last_cell_metrics["statements"]]
    assert statuses == ["RESTORED"], statuses
    assert "NOTEBOOK-ITEMS-UNREAD" not in [getattr(w.message, "code", None) for w in caught]
