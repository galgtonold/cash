"""A ``functools.wraps`` decorator that counts or logs calls into a dict
(the classic ``counted`` / ``log_calls`` decorator) counts on every Run All
and after a restart, as in a plain kernel: called by name or handed to
``s.apply``. The analysis read only the wrapped function's body, so the
decorated call was served from the cache and the counts stayed empty.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

ROWS = 20
CELLS = [
    f"import time, functools, pandas as pd, numpy as np\ns = pd.Series(np.arange({ROWS}))",
    "calls = {}\ndef counted(fn):\n    @functools.wraps(fn)\n    def w(*a):\n"
    "        calls[fn.__name__] = calls.get(fn.__name__, 0) + 1\n        return fn(*a)\n    return w\n"
    "@counted\ndef dbl(v):\n    time.sleep(0.3); return v * 2\n"
    "@counted\ndef tpl(v):\n    time.sleep(0.015); return v * 3",
    "y = dbl(3)",
    "r = s.apply(tpl)",
    "print(calls)",
]
WANT = repr({"dbl": 1, "tpl": ROWS})


def test_the_counts_are_made_again_on_every_run(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    for _ in range(2):
        nb_runner.run_all()
        assert nb_runner.peek("calls") == WANT
    nb_runner.restart()
    nb_runner.run_all()
    assert nb_runner.peek("calls") == WANT
