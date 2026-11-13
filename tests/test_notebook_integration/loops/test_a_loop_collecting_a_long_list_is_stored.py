"""A loop collecting a long list of Timedeltas or of parsed records is stored.

``deltas += list(group['ts'].diff().dropna())`` over a grouped frame, and
``results.append(parse(line))`` over a file, each fill a list of more
objects than the identity walk counted. It gave up ("'deltas' holds more
objects than cash looks through"), the loop ran again on every run, and so
did what read the list next. With a class of the notebook's keeping state,
the check for the classes a method call may reach gave up on the list too.
"""

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.loops, pytest.mark.timeout(600)]

TIMES = [
    "%cash_badge print\nimport numpy as np, pandas as pd\n"
    "rng = np.random.default_rng(0)\n"
    "df = pd.DataFrame({'a': rng.integers(0, 300, 1_200_000), 't': rng.random(1_200_000).cumsum()})\n"
    "df['ts'] = pd.to_datetime(df['t'] * 1e9)\n"
    "class Tracker:\n"
    "    seen = []",
    "deltas = []\nfor _, group in df.groupby('a'):\n    deltas += list(group['ts'].diff().dropna())",
    "q = np.quantile(np.array(deltas, dtype='timedelta64[ns]').astype('int64'), [0.1, 0.5, 0.9])",
]

RECORDS = [
    "%cash_badge print\n"
    "with open('lines.txt', 'w') as fh:\n"
    "    fh.write(''.join(f'{i},{i * 0.5},name{i}\\n' for i in range(300_000)))\n"
    "def parse(line):\n"
    "    a, b, c = line.rstrip().split(',')\n"
    "    return {'a': int(a), 'b': float(b), 'c': c}",
    # Each pass slow enough, all of them alike, for the share check to have
    # time for 300,000 dicts: one slow pass would split the loop.
    "results = []\nfor line in open('lines.txt'):\n    results.append(parse(line))\n    sum(map(len, [line] * 400))",
]


def test_a_loop_collecting_timedeltas_is_served_and_so_is_what_reads_them(nb_runner):
    nb_runner.create_notebook(TIMES)
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = nb_runner.peek("(len(deltas), q.tolist())")
    nb_runner.run_all()
    loop, reader = nb_runner.get_raw_output(2), nb_runner.get_raw_output(3)
    assert "  CACHED: for _, group in df.groupby('a'):" in loop, loop
    assert "looks through" not in loop, loop
    assert shows_cached(reader), reader
    assert nb_runner.peek("(len(deltas), q.tolist())") == first
    assert nb_runner.peek("type(deltas[0]).__name__") == "'Timedelta'"


def test_a_loop_collecting_parsed_records_is_served(nb_runner):
    nb_runner.create_notebook(RECORDS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    loop = nb_runner.get_raw_output(2)
    assert "  CACHED: for line in open('lines.txt'):" in loop, loop
    assert "looks through" not in loop, loop
    assert nb_runner.peek("(len(results), results[-1])") == "(300000, {'a': 299999, 'b': 149999.5, 'c': 'name299999'})"
