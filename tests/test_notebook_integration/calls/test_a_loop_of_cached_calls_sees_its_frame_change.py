"""A loop of cached calls over a big frame sees the frame change in place.

The digest of a frame handed to the calls of a loop is now kept while
copy-on-write proves the frame unchanged, instead of being read again for
every call and every check around it (p04-backtest: 0.7 s a call). A kept
digest must never go stale: a frame written in place through pandas, or
through a handle pandas gave out, between two runs of the loop gives the
calls a new key, and they run again with what the frame now holds.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = "import cash\n%load_ext cash\n%cash_badge print\n%cash_on"
MODULE = """import cash


@cash.cache
def signal(close, k):
    return close.shift(k)


@cash.cache
def backtest(close, signal, mask, n):
    return float(close.iloc[-1, 0] + signal.iloc[-1, 0] * n + mask.iloc[-1, 0])
"""
LOOP = (
    "rows = []\n"
    "for k in [1, 2]:\n"
    "    s = btloop_frames.signal(close, k)\n"
    "    for n in [1, 2]:\n"
    "        rows.append(btloop_frames.backtest(close, s, mask, n))\n"
    "print('R', rows)"
)


def test_a_frame_written_in_place_gives_the_calls_new_keys(nb_runner):
    (nb_runner.work_dir / "btloop_frames.py").write_text(MODULE, encoding="utf-8")
    nb_runner.create_notebook(
        [
            SETUP,
            "import numpy as np, pandas as pd, btloop_frames\n"
            "close = pd.DataFrame(np.ones((2000, 200)))  # 3.2 MB\n"
            "mask = pd.DataFrame(np.ones((2000, 200)), dtype=float) > 0",
            "edits = 0",
            LOOP,
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R [3.0, 4.0, 3.0, 4.0]" in nb_runner.get_output(4)
    nb_runner.run_cells([4])
    assert "R [3.0, 4.0, 3.0, 4.0]" in nb_runner.get_output(4)
    nb_runner.set_cell_source(3, "close.iloc[-1, 0] = 5.0\nmask.iloc[-1, 0] = False")
    nb_runner.run_cells([3, 4])
    assert "R [6.0, 7.0, 6.0, 7.0]" in nb_runner.get_output(4), "a write through pandas went unseen"
    nb_runner.set_cell_source(3, "close.iloc[-1, 0] = 5.0\nmask.iloc[-1, 0] = False\nclose[0].array[-1] = 7.0")
    nb_runner.run_cells([3, 4])
    assert "R [8.0, 9.0, 8.0, 9.0]" in nb_runner.get_output(4), "a write through .array went unseen"


def test_a_frame_the_loop_itself_writes_gives_each_call_its_own_key(nb_runner):
    (nb_runner.work_dir / "btloop_frames.py").write_text(MODULE, encoding="utf-8")
    nb_runner.create_notebook(
        [
            SETUP,
            "import numpy as np, pandas as pd, btloop_frames\n"
            "close = pd.DataFrame(np.ones((2000, 200)))\n"
            "mask = pd.DataFrame(np.ones((2000, 200)), dtype=float) > 0",
            "out = []\n"
            "for i in range(3):\n"
            "    close.iloc[-1, 0] = 10.0 + i\n"
            "    out.append(btloop_frames.backtest(close, close, mask, 0))\n"
            "print('O', out, close.iloc[-1, 0])",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "O [11.0, 12.0, 13.0] 12.0" in nb_runner.get_output(3)
    nb_runner.run_all()
    assert "O [11.0, 12.0, 13.0] 12.0" in nb_runner.get_output(3)
