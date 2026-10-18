"""A statement that looks a function up by a string and calls it at once --
``globals()[name](3)``, ``NS.get(name)(3)``, ``getattr(me, name)(3)`` -- is
served on its trace on Python 3.12+, where before it ran every time.

Its key cannot name the function the string finds, so the run's trace
records what was called, the name holding it, and what the function read,
and a hit is served only when the trace says none of it changed
(``cache_key.lookups_followed``). A value read by a string and used as data
records nothing, so that statement still runs every time, as it does before
3.12 whatever the lookup.
"""

from __future__ import annotations

import sys
import warnings
from unittest.mock import patch

import pytest

from cash.notebook.cache_status import CacheStatus

TRACED = sys.version_info >= (3, 12)

DEFINITIONS = """import time
cfg = 1
name = 'make'
NS = globals()
def make(x):
    time.sleep(0.05)
    return x + cfg
"""

CALLED = {
    "globals-in-the-statement": ("pass", "r = globals()[name](3)"),
    "a-name-holding-the-namespace": ("pass", "r = NS[name](3)"),
    "globals-in-a-helper": ("def g():\n    return globals()[name](3)", "r = g()"),
    "a-comprehension": ("pass", "r = [globals()[n](3) for n in (name,)][0]"),
}

COMPUTED, RESTORED = CacheStatus.COMPUTED, CacheStatus.RESTORED


def _run(magics, cells):
    """Run *cells* in order as one Run All; the statuses of the last one's statements."""
    statuses = []
    with (
        patch("cash.notebook.upstream.checker.get_notebook_cells", side_effect=lambda _path=None: list(cells)),
        patch("cash.notebook.upstream.checker.get_notebook_cells_with_ids", return_value=None),
    ):
        for cell in cells:
            result = magics._cell_executor.execute_cell(cell, ttl=None, cell_id=magics.resolve_cell_id())
            magics._finalize_cell_body(cell, result)
            last = cells[-1].splitlines()[-1]
            statuses = [CacheStatus.parse(m.get("status")) for m in result.all_metrics if m.get("code") == last]
    return statuses


@pytest.fixture
def run(cash_magics, monkeypatch):
    monkeypatch.delenv("CASH_TRACE", raising=False)
    cash_magics.cash_on("")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        yield lambda cells: _run(cash_magics, cells)


@pytest.mark.parametrize("case", list(CALLED))
def test_it_is_served_and_reruns_when_what_it_found_reads_changes(run, mock_shell, case):
    helper, statement = CALLED[case]
    cells = [DEFINITIONS, helper, statement]
    assert run(cells) == [COMPUTED]
    assert run(cells) == ([RESTORED] if TRACED else [COMPUTED])
    assert mock_shell.user_ns["r"] == 4
    assert run([DEFINITIONS.replace("cfg = 1", "cfg = 2"), helper, statement]) == [COMPUTED]
    assert mock_shell.user_ns["r"] == 5


def test_it_reruns_when_the_name_holds_another_function(run, mock_shell):
    cells = [DEFINITIONS, "pass", "r = globals()[name](3)"]
    run(cells)
    assert run(cells) == ([RESTORED] if TRACED else [COMPUTED])
    swapped = DEFINITIONS + "def make(x):\n    return x * 100\n"
    assert run([swapped, "pass", "r = globals()[name](3)"]) == [COMPUTED]
    assert mock_shell.user_ns["r"] == 300


def test_it_reruns_when_the_name_holds_another_library_function(run, mock_shell):
    """A library's function has no code of the user's to record: the name
    holding it is what the trace checks."""
    statement = "# @cash:persist\nr = globals()[name](3)"
    cells = ["from operator import neg as op\nname = 'op'", statement]
    assert run(cells) == [COMPUTED]
    assert run(cells) == ([RESTORED] if TRACED else [COMPUTED])
    assert mock_shell.user_ns["r"] == -3
    assert run(["from operator import pos as op\nname = 'op'", statement]) == [COMPUTED]
    assert mock_shell.user_ns["r"] == 3


@pytest.mark.parametrize(
    "statement",
    [
        "r = globals()['cfg'] + 1",
        "f = globals()[name]",
        "r = eval('make(3)')",
    ],
)
def test_a_value_read_by_a_string_still_runs_every_time(run, statement):
    cells = [DEFINITIONS, statement]
    assert run(cells) == [COMPUTED]
    assert run(cells) == [COMPUTED]
