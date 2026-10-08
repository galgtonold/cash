"""``df['a'] = feat(df)`` stays cached once ``df = load()`` was served.

The statement entry of ``df = load()`` refers to the call's entry
(``call_refs``); restoring it reads the call's value back. The dict of values
read back was kept alive by a reference cycle until the next cyclic
collection, and the share check of the next statement counted it as another
holder of ``df``: the feature statement was refused ("holds an object another
variable or container also holds") and ``feat`` ran on every Run All.
"""

from __future__ import annotations

import gc
import os

import pytest

from cash.notebook.call_refs import CallRef, resolve_call_refs
from tests._cell_driver import run_cash_cell

pd = pytest.importorskip("pandas")

DEFS = (
    "import os, time\n"
    "import pandas as pd\n"
    "def load():\n"
    "    time.sleep(0.25)\n"
    "    return pd.DataFrame({'x': [float(i) for i in range(1000)]})\n"
    "def feat(d):\n"
    "    os.write(FD, b'f')\n"
    "    time.sleep(0.25)\n"
    "    return d.x * 3\n"
)
CELLS = ["df = load()", "df['a'] = feat(df)"]


def test_feat_runs_once_over_three_run_alls(cash_magics, mock_shell, clean_backend, tmp_path):
    log = tmp_path / "calls.log"
    fd = os.open(log, os.O_WRONLY | os.O_CREAT | os.O_APPEND)
    try:
        mock_shell.user_ns["__name__"] = "__main__"
        mock_shell.user_ns["FD"] = fd
        run_cash_cell(cash_magics, DEFS)
        for _ in range(3):
            for code in CELLS:
                run_cash_cell(cash_magics, code, cells=[DEFS, *CELLS])
        assert list(mock_shell.user_ns["df"].a[:3]) == [0.0, 3.0, 6.0]
    finally:
        os.close(fd)
    assert log.read_text() == "f", f"feat ran {len(log.read_text())} times over three Run Alls"


class _Backend:
    def __init__(self, value):
        self.value = value

    def get(self, key):
        return {"value_digest": "d"}, self.value


def test_resolving_a_reference_keeps_no_dict_of_the_values_read():
    """Asked with the cyclic collector off, as between two statements: a
    dict of the loaded values still alive is a holder of each of them."""
    value = pd.DataFrame({"x": [1.0, 2.0]})
    payload = {"call_refs": True, "variables": {"df": CallRef("call:k", "d")}}
    gc.disable()
    try:
        resolved = resolve_call_refs(payload, _Backend(value))
        assert resolved["variables"]["df"] is value
        dicts = [r for r in gc.get_referrers(value) if isinstance(r, dict)]
        assert dicts == [resolved["variables"]], [list(d)[:3] for d in dicts]
    finally:
        gc.enable()
