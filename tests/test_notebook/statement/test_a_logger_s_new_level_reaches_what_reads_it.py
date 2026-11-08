"""A statement reading a logger, or a value holding one, sees its new level.

A logger pickles by name (``getLogger(name)``), so a digest of its pickle
held nothing of its level. ``log.setLevel(10)`` was taken to change
nothing, and ``seen = level_of(cfg)`` -- ``cfg`` holding the logger three
dicts deep -- was served 30 where plain Python gives 10; so was
``seen = level_of(log)``.
"""

from __future__ import annotations

import logging

import pytest

from tests._cell_driver import run_cash_cell


@pytest.fixture
def no_root_handlers(monkeypatch):
    # pytest's own handler on the root logger holds a stream, which no key
    # reads: the logger is then keyed without it, as in a notebook.
    monkeypatch.setattr(logging.root, "handlers", [])


@pytest.mark.parametrize("reader", ["level_of(cfg)", "level_of({'a': {'b': {'c': {'log': log}}}})"])
def test_a_reader_runs_again_after_set_level(cash_magics, cash_instance, no_root_handlers, tmp_path, reader):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    calls = tmp_path / "calls"
    setup = (
        "import logging\n"
        "log = logging.getLogger('cash_test_logger_level')\n"
        "log.setLevel(30)\n"
        "cfg = {'a': {'b': {'c': {'log': log}}}}\n"
        f"CALLS = {str(calls)!r}\n"
        "def level_of(c):\n"
        "    with open(CALLS, 'a') as f:  # @cash:assume-safe\n"
        "        f.write('x')\n"
        "    return c['a']['b']['c']['log'].level"
    )
    cells = [
        setup,
        f"seen = {reader}",
        # Fills the logger's cache of enabled levels: no change to its state.
        "log.warning('reading')",
        f"# served\nseen = {reader}",
        "log.setLevel(10)",
        f"# again\nseen = {reader}",
    ]
    for cell in cells:
        run_cash_cell(cash_magics, cell, cells=cells)
    assert cash_magics.shell.user_ns["seen"] == 10
    assert calls.read_text() == "xx"
