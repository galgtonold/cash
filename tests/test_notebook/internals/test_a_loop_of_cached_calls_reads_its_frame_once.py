"""A loop handing one big frame to a module's cached functions reads it a fixed number of times.

p04-backtest: ``for lb in ...: s = signal(close, ...); for n in ...: for
hold in ...: r = run_backtest(close, s, mask, ...)`` -- 32 calls, each
given the same 50 MB ``close`` and ``mask``. The loop's in-place-change
watch fingerprinted every frame handed to the calls before and after each
inner loop, and the value checks around each call read them again: 0.7 s
a call, 47 s against 26 s plain. The frames' digests are now kept while
copy-on-write proves them unchanged (`FrameDigests`), so the reads do not
grow with the number of calls.
"""

from __future__ import annotations

import importlib
import os
import sys

import pytest

from cash import content_hashers
from tests._cell_driver import run_cash_cell

pd = pytest.importorskip("pandas")

if int(pd.__version__.split(".")[0]) < 3:  # pragma: no cover - the memo needs copy-on-write
    pytest.skip("copy-on-write is pandas 3's only mode", allow_module_level=True)

MODULE = """
import cash
import numpy as np
import pandas as pd


@cash.cache
def signal(close, skip):
    return close.shift(skip)


@cash.cache
def backtest(close, signal, mask, n, hold):
    return float(signal.iloc[-1, 0]) + n + hold
"""


@pytest.fixture
def lib(tmp_path, monkeypatch):
    name = f"_loop_lib_{os.getpid()}_{id(tmp_path)}"
    (tmp_path / f"{name}.py").write_text(MODULE, encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    yield importlib.import_module(name).__name__
    sys.modules.pop(name, None)


@pytest.fixture
def reads(monkeypatch, mock_shell):
    """Full reads of the frames named ``close`` and ``mask``."""
    seen = []
    real = content_hashers.hash_pandas

    def counting(value):
        if any(value is mock_shell.user_ns.get(name) for name in ("close", "mask")):
            seen.append(1)
        return real(value)

    monkeypatch.setattr(content_hashers, "hash_pandas", counting)
    return seen


def test_the_frames_are_not_read_again_per_call(cash_magics, mock_shell, lib, reads):
    mock_shell.user_ns["__name__"] = "__main__"
    run_cash_cell(
        cash_magics,
        f"import numpy as np, pandas as pd\nimport {lib} as lib\n"
        "close = pd.DataFrame(np.random.default_rng(0).random((3000, 400)) + 1)  # 9.6 MB, mask 1.2 MB\n"
        "mask = close > 1.1\n",
    )
    reads.clear()
    run_cash_cell(
        cash_magics,
        "rows = []\n"
        "for skip in [1, 2, 3]:\n"
        "    s = lib.signal(close, skip)\n"
        "    for n in [1, 2]:\n"
        "        for hold in [1, 2]:\n"
        "            rows.append(lib.backtest(close, s, mask, n, hold))\n",
    )
    assert len(mock_shell.user_ns["rows"]) == 12
    # Was 44 for these 15 calls: about three reads of each frame per call.
    assert len(reads) <= 4, f"close and mask were read {len(reads)} times for 15 calls"
