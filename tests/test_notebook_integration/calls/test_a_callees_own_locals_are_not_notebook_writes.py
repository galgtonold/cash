"""A name a callee binds for itself is not a notebook variable it writes.

``acc`` below is local to ``collect`` (an annotated assignment) and ``r`` is
its loop variable. A notebook holding globals of the same names must not see
``x = collect(data)`` as writing them: that made the statement run every time
and bumped the lineage of the unrelated globals.
"""

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration]

SETUP = "import cash\n%cash_on\n%cash_badge print"


def test_a_statement_calling_it_still_caches(nb_runner):
    nb_runner.create_notebook(
        [
            SETUP,
            "acc = ['notebook']\nr = ['notebook']\ndata = [1, 2, 3]",
            "import time\n"
            "def collect(rows):\n"
            "    acc: list = []\n"
            "    for r in rows:\n"
            "        acc.append(r * 2)\n"
            "    time.sleep(0.15)\n"
            "    return acc",
            "x = collect(data)",
        ],
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.run_all()
    assert shows_cached(nb_runner.get_output(4)), nb_runner.get_output(4)
    assert nb_runner.peek("x") == "[2, 4, 6]"
    assert nb_runner.peek("acc") == "['notebook']"
