"""A call in a loop reads the library routine its callee swapped in.

``step`` reads ``m.get`` and then sets it to ``logging.getLogger``: the
second call of ``[step(n) for n in [name, name]]`` reads the logger's level
in plain Python. Keyed alike, it was served the first call's answer, the
``-1`` the routine swapped out gave, and so was a rerun of the cell. A call
is keyed on which routine a module attribute holds, a library routine by
its name, so the second call is another key.

Only the swap is under test here; what a change to the logger's own level
does to its readers is not.
"""

from __future__ import annotations

import logging

import pytest

from tests._cell_driver import run_cash_cell

LOGGER = "zz.cash.swapped.in"
SETUP = (
    f"import logging, types\nlogging.getLogger({LOGGER!r}).setLevel(20)\n"
    "m = types.ModuleType('m')\nclass Fake:\n    level = -1\ndef fake(n):\n    return Fake\n"
)
SWAPS = {
    "setattr": "setattr(m, 'get', {new})",
    "assigned": "m.get = {new}",
}


def _cell(swap: str) -> str:
    return (
        f"{swap.format(new='fake')}\n"
        f"def step(n):\n    v = m.get(n).level\n    {swap.format(new='logging.getLogger')}\n    return v\n"
        f"lv = [step(n) for n in [{LOGGER!r}, {LOGGER!r}]]"
    )


@pytest.fixture(autouse=True)
def _store_everything(cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    yield
    logging.getLogger(LOGGER).setLevel(logging.NOTSET)


@pytest.mark.parametrize("swap", sorted(SWAPS))
def test_the_second_call_reads_the_logger_on_every_run(cash_magics, mock_shell, swap):
    cell = _cell(SWAPS[swap])
    cells = [SETUP, cell]
    run_cash_cell(cash_magics, SETUP, cells=cells)
    for _ in range(2):
        run_cash_cell(cash_magics, cell, cells=cells)
        assert mock_shell.user_ns["lv"] == [-1, 20]
