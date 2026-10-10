"""A long loop that draws fresh values through a helper re-runs what reads it.

A loop run as one unit names the lists it fills by its key when its outcome
is a function of that key. Re-running such a loop that stamps the clock, makes
a uuid or draws ``next()`` from a global counter in a helper (or calls
``os.urandom`` inline) left new values under the old lineage, and the
expensive cell below was restored with the result for the old values. The
helpers' bodies are read now, and a global iterator a helper draws from
counts, so those lists are named by their values.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.loops, pytest.mark.timeout(240)]

SUMMARY_DEF = "import time\ndef summarize(v):\n    time.sleep(0.3)\n    return sum(v)\n"


@pytest.mark.parametrize(
    "helper, body",
    [
        ("def stamp(i):\n    return int(time.time() * 1e6) % 1000", "out.append(stamp(i))"),
        ("import itertools\nids = itertools.count(1)\ndef new_id(i):\n    return next(ids)", "out.append(new_id(i))"),
        ("import uuid\ndef tag(i):\n    return uuid.uuid4().int % 1000", "out.append(tag(i))"),
        ("import os", "out.append(os.urandom(1)[0])"),
    ],
    ids=["the clock", "a global counter", "a uuid", "os.urandom"],
)
def test_the_cell_below_sees_the_new_values(nb_runner, helper, body):
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on",
            SUMMARY_DEF + helper,
            "out = []",
            f"for i in range(300):\n    {body}",
            "s = summarize(out)\nprint(s)",
        ]
    )
    nb_runner.start_kernel()
    nb_runner.run_all()
    first = nb_runner.peek("out")
    nb_runner.run_cells([3, 4, 5])
    assert nb_runner.peek("out") != first, "positive control: the loop left new values"
    expected = nb_runner.peek("sum(out)").strip()
    assert nb_runner.get_output(5).strip() == expected, (
        "the summary was restored for the old values: " + nb_runner.get_raw_output(5)
    )
    assert nb_runner.peek("s").strip() == expected
