"""A logger's level set in a cell above reaches the statements that read it.

``logging.getLogger('app').setLevel(40)`` changes state the ``logging``
module keeps, as ``pd.set_option(...)`` does for pandas. A setter was told
by its name, ``set`` or ``set_*``, so logging's camelCase ``setLevel`` was
not one: ``logging`` kept its lineage, and a reader run before the setting
cell was served the old level after it -- read directly, or through a
routine a module attribute holds (``m.get = logging.getLogger`` in a loop).

The notebook's cells are in the order given; correct is what running them
top to bottom gives, so a setting in a cell below a reader does not reach it.
"""

from __future__ import annotations

import logging

import pytest

from tests._cell_driver import run_cash_cell

LOGGER = "zz.cash.level.above"
SETUP = (
    f"import logging, types\nlogging.getLogger({LOGGER!r}).setLevel(20)\n"
    "m = types.ModuleType('m')\nclass Fake:\n    level = -1\ndef fake(n):\n    return Fake\n"
)
SETTING = f"logging.getLogger({LOGGER!r}).setLevel(40)"

READERS = {
    "direct": f"lv = logging.getLogger({LOGGER!r}).level",
    "through a function": f"def level(n):\n    return logging.getLogger(n).level\nlv = level({LOGGER!r})",
    "through a module attribute swapped in a loop": (
        "setattr(m, 'get', fake)\n"
        "def step(n):\n    v = m.get(n).level\n    setattr(m, 'get', logging.getLogger)\n    return v\n"
        f"lv = [step(n) for n in [{LOGGER!r}, {LOGGER!r}]][1]"
    ),
}


@pytest.fixture(autouse=True)
def _reset_level(cash_instance):
    cash_instance.config.min_execution_time_to_cache_seconds = 0.0
    cash_instance.config.call_cost_floor_seconds = 0.0
    yield
    logging.getLogger(LOGGER).setLevel(logging.NOTSET)


def _run_all(cells: list[str]):
    logging.getLogger(LOGGER).setLevel(logging.NOTSET)
    ns: dict = {}
    for cell in cells:
        exec(cell, ns)
    logging.getLogger(LOGGER).setLevel(logging.NOTSET)
    return ns["lv"]


@pytest.mark.parametrize("above", [True, False], ids=["setting above", "setting below"])
@pytest.mark.parametrize("reader", sorted(READERS))
def test_the_reader_run_again_after_the_setting_reads_what_run_all_does(cash_magics, mock_shell, reader, above):
    body = READERS[reader]
    cells = [SETUP, SETTING, body] if above else [SETUP, body, SETTING]
    want = _run_all(cells)
    for cell in (SETUP, body, SETTING, body):
        run_cash_cell(cash_magics, cell, cells=cells)
    assert mock_shell.user_ns["lv"] == want
