"""A slow statement that draws from an iterator made in an earlier cell.

``header = parse(next(rows))`` moves ``rows``, but the stored entry holds only
``header``. Served from the cache on the second Run All, it left ``rows`` where
the cell above put it, and the next cell read the header a second time. Such a
statement runs every time; the slow call inside it is still cached.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]

SETUP = (
    "import time, io, csv, itertools\n"
    "def parse(x):\n"
    "    time.sleep(0.2)\n"
    "    return x\n"
    "LINES = ['h1,h2', '1,2', '3,4']"
)


@pytest.mark.parametrize(
    "cells, expected",
    [
        (["rows = iter(LINES)", "header = parse(next(rows))", "r = list(rows)"], ["1,2", "3,4"]),
        (["rd = csv.reader(LINES)", "header = parse(next(rd))", "r = list(rd)"], [["1", "2"], ["3", "4"]]),
        (["buf = io.StringIO('\\n'.join(LINES))", "header = parse(buf.readline())", "r = buf.read()"], "1,2\n3,4"),
        (
            ["src = iter(range(10))", "batch = parse(list(itertools.islice(src, 3)))", "r = list(itertools.islice(src, 3))"],
            [3, 4, 5],
        ),
    ],
    ids=["iter", "csv.reader", "StringIO", "islice"],
)
def test_the_next_reader_continues_after_a_second_run_all(nb_runner, cells, expected):
    nb_runner.create_notebook([SETUP, *cells])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("r") == repr(expected), "first Run All"
    nb_runner.run_all()
    assert nb_runner.peek("r") == repr(expected), "second Run All"
