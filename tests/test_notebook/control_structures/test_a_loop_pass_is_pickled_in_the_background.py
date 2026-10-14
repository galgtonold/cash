"""Each pass of a loop is stored without pickling on the cell's thread.

``for k in ...: h, _ = run(data, k); grid[k] = score(h)`` stores every
pass's result, so an edit inside the loop recomputes only what it changes.
A pass's frame the RAM tier keeps no copy of its own -- refused, over its
eviction target, or copied by pandas' deep copy, as a frame of Python
strings is without pyarrow (p28) -- was pickled on the cell's thread:
0.2 s a pass for p28's 900k rows, seconds for a bigger frame. It is copied
on that thread and pickled in the background now, as one the RAM tier
keeps is, and nothing the cell does to it later reaches the entry. A
``# @cash:persist`` statement still lands on disk.
"""

from __future__ import annotations

import os
import sys
import threading

import pandas as pd
import pytest

from cash.backends import FileBackend, InMemoryBackend, TieredBackend, serialization
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

MiB = 1 << 20
ROWS = 400_000  # 6+ MB a frame: over the RAM tier's eviction target below

LIB = (
    "import time\n"
    "import numpy as np\n"
    "import pandas as pd\n"
    "def run(k):\n"
    f"    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
    f"    n = {ROWS}\n"
    "    return pd.DataFrame({'x': np.arange(n, dtype=float) * k,\n"
    "                         's': pd.Series(['t%d' % (i % 97) for i in range(n)], dtype=TEXT)}), {'k': k}\n"
)

#: (text dtype of the frame, RAM tier cap): refused by the cap, or copied
#: by pandas (Python strings) and so not the RAM tier's own.
CASES = {"refused": ("str", 4 * MiB), "python-strings": ("string[python]", None)}


@pytest.fixture(params=sorted(CASES))
def notebook(request, cash_magics, cash_instance, mock_shell, tmp_path, monkeypatch):
    """The notebook over a RAM and a disk tier; a module ``lib`` with the
    slow ``run``; and the threads that pickled a frame for a store."""
    text, cap = CASES[request.param]
    ram, disk = InMemoryBackend(max_size_bytes=cap), FileBackend(str(tmp_path / "cache"), flush_interval=0)
    cash_instance.backend = TieredBackend([ram, disk])
    name = f"_loop_pass_lib_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(f"TEXT = {text!r}\n" + LIB, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    pickled_on: list[str] = []
    split = serialization.PickleSerializer.serialize_split

    def spy(self, data, *, copy=True):
        if _holds_frame(data):
            pickled_on.append(threading.current_thread().name)
        return split(self, data, copy=copy)

    monkeypatch.setattr(serialization.PickleSerializer, "serialize_split", spy)
    run_cash_cell(cash_magics, f"import {name} as lib")
    yield cash_magics, disk, pickled_on
    disk._writes.wait_all()
    disk.shutdown()
    sys.modules.pop(name, None)


def _holds_frame(value, depth=0) -> bool:
    if isinstance(value, pd.DataFrame):
        return True
    if depth < 3 and isinstance(value, (tuple, list)):
        return any(_holds_frame(item, depth + 1) for item in value)
    if depth < 3 and isinstance(value, dict):
        return any(_holds_frame(item, depth + 1) for item in value.values())
    return False


def _stored_frames(disk) -> list[pd.DataFrame]:
    disk._writes.wait_all()
    found: list[pd.DataFrame] = []
    for entry in disk.list_entries() or ():
        _meta, value = disk.get(entry["key"])
        stack = [value]
        while stack:
            item = stack.pop()
            if isinstance(item, pd.DataFrame):
                found.append(item)
            elif isinstance(item, (tuple, list)):
                stack.extend(item)
            elif isinstance(item, dict):
                stack.extend(item.values())
    return found


def test_each_pass_is_pickled_off_the_cells_thread(notebook):
    magics, disk, pickled_on = notebook
    run_cash_cell(magics, "grid = {}\nfor k in [1, 2, 3]:\n    h, _ = lib.run(k)\n    grid[k] = float(h['x'].sum())")
    disk._writes.wait_all()
    assert pickled_on, "no pass was written: the test checks nothing"
    assert threading.current_thread().name not in pickled_on, "a pass was pickled on the cell's thread"
    run_cash_cell(magics, "h['s'].array[1] = 'changed'\nh.loc[0, 'x'] = -1.0\nh['x'] = h['x'] * 0")
    stored = _stored_frames(disk)
    assert stored, "no pass reached the disk"
    for frame in stored:
        assert frame["x"].iloc[1] != 0.0, "an edit after the loop reached a stored pass"
        assert frame["x"].iloc[0] == 0.0
        assert frame["s"].iloc[1] == "t1"


def test_a_persist_statement_lands_on_disk(notebook):
    magics, disk, pickled_on = notebook
    run_cash_cell(magics, "# @cash:persist\nh, _ = lib.run(2)")
    assert any(frame["x"].iloc[1] == 2.0 for frame in _stored_frames(disk))
    assert threading.current_thread().name not in pickled_on
