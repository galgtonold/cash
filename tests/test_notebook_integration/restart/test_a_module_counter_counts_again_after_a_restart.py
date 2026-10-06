"""A module's counter is where a top-to-bottom run leaves it after a restart.

``metrics.increment(5)`` in a cell, the counter kept in ``metrics.py``: after
a kernel restart, running the notebook served the cell from the cache, so
``increment`` never ran and the counter stayed at 0 where a plain kernel has
5. The statement sets state on the module, which no restore puts back; it
runs again.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.restore, pytest.mark.timeout(240)]

LIB = (
    "import time\n"
    "COUNT = 0\n"
    "def _bump(n):\n    global COUNT\n    COUNT += n\n"
    "def increment(n):\n    global COUNT\n    time.sleep(0.2)\n    COUNT += n\n    return COUNT\n"
    "def increment_via_helper(n):\n    time.sleep(0.2)\n    _bump(n)\n    return n\n"
)


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("metrics.increment(5)", 5),
        ("metrics.increment_via_helper(5)", 5),
        ("metrics.increment(5)\nmetrics.increment(5)", 10),
        ("for _ in range(2):\n    metrics.increment(5)", 10),
    ],
    ids=["module_function", "through_a_helper", "two_calls", "loop"],
)
def test_the_counter_after_a_restart(nb_runner, tmp_path, body, expected):
    (tmp_path / "metrics.py").write_text(LIB, encoding="utf-8")
    nb_runner.create_notebook(["import metrics", body, "print('T', metrics.COUNT)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert f"T {expected}" in nb_runner.get_output(3), nb_runner.get_raw_output(3)

    nb_runner.restart()
    nb_runner._init_cash()
    nb_runner.run_all()

    assert nb_runner.peek("metrics.COUNT") == str(expected), nb_runner.get_raw_output(2)
