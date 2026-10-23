"""Cash pickles a statement's values to hash and store them, and the
pickle runs the value's own ``__reduce__``: code of the user's that plain
Python never runs there. One that writes into a frame sharing other
frames' cells (``df.at[3, 'o'] = ...``) changed ``df`` behind every
statement, and a reader of ``df['o']`` below was served what it read
before the write. Cash now does not run such a reducer: it does not
pickle the value, and the notebook gives what plain Python gives, where
the reducer never runs. A ``__getstate__`` that only reads stays cached.
"""

from __future__ import annotations

import ast

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.mutations, pytest.mark.timeout(240)]

SETUP = "import cash\n%cash_on\nimport time\nimport numpy as np, pandas as pd"
FRAME = (
    "df = pd.DataFrame({'x': np.arange(100_000.), 'o': [{'i': i} for i in range(100_000)]})\n"
    "d2 = df[df.x >= 0]\n"
    "d3 = df[df.x % 2 == 0]"
)
WRITER = (
    "class W:\n"
    "    def __reduce__(self):\n"
    "        df.at[3, 'o'] = {{'i': 3, 'z': {z}}}\n"
    "        return (W, ())\n"
    "w = (time.sleep(0.3), W())[1]"
)
READER = "g = (time.sleep(0.3), df['o'].map(str).tolist())[1]"
CHECK = "print('R', g[3], df['o'].map(str).tolist()[3])"
STATUSES = (
    "[str(s.get('status')) for s in "
    "get_ipython().magics_manager.magics['line']['cash_status'].__self__._last_cell_metrics['statements']]"
)


def test_the_reader_below_is_never_served_what_the_reduce_changed(nb_runner):
    nb_runner.create_notebook([SETUP, FRAME, WRITER.format(z=1), READER, CHECK])
    nb_runner.start_kernel()
    got = []
    for z, restart in ((1, False), (2, False), (2, True)):
        if restart:
            nb_runner.restart()
            nb_runner._init_cash()
        nb_runner.set_cell_source(3, WRITER.format(z=z))
        nb_runner.run_all()
        got.append(nb_runner.get_output(5).strip())
    # Plain Python never pickles ``w``, so its reducer never writes ``df``.
    assert got == ["R {'i': 3} {'i': 3}"] * 3, got


def test_a_getstate_that_only_reads_stays_cached(nb_runner):
    holder = (
        "class P:\n"
        "    def __init__(self):\n"
        "        self.rows = list(range(1000))\n"
        "    def __getstate__(self):\n"
        "        state = self.__dict__.copy()\n"
        "        state.pop('cache', None)\n"
        "        return state\n"
        "p = (time.sleep(0.3), P())[1]"
    )
    nb_runner.create_notebook([SETUP, FRAME, holder])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    statuses = ast.literal_eval(nb_runner.peek(STATUSES))
    assert statuses[-1] == "RESTORED", statuses
