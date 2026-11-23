"""A frame a statement hands on only as a part (``t = f(x[x.a >= 0])``) is
still seen when the statement changes it some other way.

Such a frame's digest is handed to no call, so it is not read before the
statement: the frame is held and checked after it (`ArgFingerprints.defer`),
and read only when that check cannot vouch for it. Each change below
reaches ``x`` through another name (``box[0]``), where no analysis of the
text sees it, and must be seen as plain Python sees it: run twice, ``x``
is bumped twice. A write of the value a cell already holds changes
nothing, and the statement is served on its second run.
"""

from __future__ import annotations

import pytest

from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")


@pytest.fixture(autouse=True)
def _copy_on_write(monkeypatch, request):
    """Copy-on-write is the only mode from pandas 3 on, opt-in before it."""
    if int(pd.__version__.split(".", 1)[0]) < 3:
        from cash.decorator import arg_hashing

        before = pd.get_option("mode.copy_on_write")
        pd.set_option("mode.copy_on_write", True)
        request.addfinalizer(lambda: pd.set_option("mode.copy_on_write", before))
        monkeypatch.setattr(arg_hashing, "_COW_PANDAS", None)


def _setup(log):
    return (
        "import os, time\n"
        "import numpy as np, pandas as pd\n"
        "box = [None]\n"
        "def ran():\n"
        f"    fd = os.open({str(log)!r}, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe\n"
        "    os.write(fd, b'.')\n"
        "    os.close(fd)\n"
        "def through_pandas(s):\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    ran()\n"
        "    f = box[0]\n"
        "    f.loc[0, 'a'] = f.loc[0, 'a'] + 1\n"
        "    return len(s)\n"
        "def through_a_handle(s):\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    ran()\n"
        "    arr = box[0]['a'].to_numpy()\n"
        "    arr.flags.writeable = True\n"
        "    arr[0] += 1\n"
        "    return len(s)\n"
        "def same_value(s):\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    ran()\n"
        "    f = box[0]\n"
        "    f.loc[0, 'a'] = f.loc[0, 'a']\n"
        "    return len(s)\n"
        "def reads_only(s):\n"
        f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    ran()\n"
        "    return len(s)\n"
    )


FRAME = "x = pd.DataFrame({'a': np.arange(300_000.0), 'b': np.ones(300_000)})\nbox[0] = x"


@pytest.mark.parametrize("fn", ["through_pandas", "through_a_handle"])
def test_a_change_made_through_another_name_is_seen(cash_magics, mock_shell, tmp_path, fn):
    mock_shell.user_ns["__name__"] = "__main__"
    log = tmp_path / "ran.log"
    run_cash_cell(cash_magics, _setup(log))
    run_cash_cell(cash_magics, FRAME)
    for _ in range(2):
        run_cash_cell(cash_magics, f"t = {fn}(x[x.b > 0])")
    assert mock_shell.user_ns["x"].loc[0, "a"] == 2.0
    assert log.read_text(encoding="utf-8") == ".."


@pytest.mark.parametrize("fn", ["same_value", "reads_only"])
def test_a_statement_that_leaves_the_frame_as_it_was_is_served(cash_magics, mock_shell, tmp_path, fn):
    mock_shell.user_ns["__name__"] = "__main__"
    log = tmp_path / "ran.log"
    run_cash_cell(cash_magics, _setup(log))
    run_cash_cell(cash_magics, FRAME)
    for _ in range(2):
        run_cash_cell(cash_magics, f"t = {fn}(x[x.b > 0])")
    assert mock_shell.user_ns["x"].loc[0, "a"] == 0.0
    assert mock_shell.user_ns["t"] == 300_000
    # Positive control for the counter: the first run ran.
    assert log.read_text(encoding="utf-8") == "."
