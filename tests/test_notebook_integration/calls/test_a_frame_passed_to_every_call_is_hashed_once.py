"""A loop of calls over the working frame hashes the frame once, not twice a call.

``scores = [evaluate(df, a) for a in alphas]`` hashed ``df`` in full before
and after each call, to see whether ``evaluate`` changed it: 0.7 s a call for
an 80 MB frame, on the first run and after every edit of the function. A frame
copy-on-write proves unchanged keeps its hash for the cell. A callee that does
change it is still seen doing so, so its call is not served from the cache.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = "import cash\n%load_ext cash\n%cash_badge print\n%cash_on"
DEFS = (
    "import time\nimport numpy as np, pandas as pd\n"
    "df = pd.DataFrame({'x': np.arange(500_000.0), 'y': np.ones(500_000)})\n"
    "def evaluate(d, alpha):\n    time.sleep(0.25)\n    return float(d.x.iloc[1]) * alpha\n"
)
COUNT = (
    "import cash.notebook.call_effects as ce\n"
    "if not hasattr(ce, 'real_hash'):\n"
    "    ce.real_hash = ce.compute_hash\n"
    "    ce.frame_hashes = []\n"
    "    def counting(value, ce=ce):\n"
    "        if type(value).__name__ == 'DataFrame':\n"
    "            ce.frame_hashes.append(1)\n"
    "        return ce.real_hash(value)\n"
    "    ce.compute_hash = counting\n"
)
HASHED = "len(__import__('cash.notebook.call_effects').notebook.call_effects.frame_hashes)"


def test_a_comprehension_and_a_loop_hash_the_frame_once_each(nb_runner):
    nb_runner.create_notebook(
        [
            SETUP,
            DEFS,
            "scores = [evaluate(df, a) for a in range(4)]",
            "out = []\nfor a in range(4, 8):\n    out.append(evaluate(df, a))",
            "print('S', scores, out)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2])
    nb_runner.peek(f"exec({COUNT!r})")
    nb_runner.run_cell(3)
    assert nb_runner.peek(HASHED) == "1"
    nb_runner.run_cell(4)
    assert nb_runner.peek(HASHED) == "2", "the loop hashed the frame again for every call"
    nb_runner.run_cell(5)
    assert "S [0.0, 1.0, 2.0, 3.0] [4.0, 5.0, 6.0, 7.0]" in nb_runner.get_output(5)


def test_a_callee_changing_the_frame_through_a_handle_is_not_served(nb_runner):
    """``d['x'].array[0] = ...`` writes past copy-on-write: the frame is
    hashed again after the call, the change is seen, and the call is not
    stored, so a later statement calling it with the same frame runs it."""
    nb_runner.create_notebook(
        [
            SETUP,
            DEFS + "def bump(d):\n    time.sleep(0.25)\n    d['x'].array[0] = d['x'].array[0] + 1\n    return 0\n",
            "r1 = [bump(df) for _ in range(1)]",
            "r2 = [bump(df) for _ in range(1)]",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("float(df.x.iloc[0])") == "2.0"
