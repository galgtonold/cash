"""A statement that runs every time runs once per Run All, and what it binds
reaches what reads it.

``# @cash:no-cache`` / ``t = globals()[name](3)`` keys its output by value.
The value's digest was filed under the statement's key, and the simulation
of the cells below builds another key for a name found by a string, so it
found no digest, took ``t`` for changed and ran the statement again for
``u``: ``make`` was called twice per Run All. A draw inside such a statement
(``NS = next(NSS)(globals())``) drew twice, so ``v`` was 99 where plain Python
gives 1. A draw that an earlier refusal caught (``t2 = Q.popleft()``, a
change to ``Q`` in place) kept its formula's lineage, so ``w`` was served the
first value on every Run All.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

HELPERS = (
    "import collections\n"
    "Q = collections.deque(range(100))\n"
    "NSS = iter([lambda g: {'x': 1}, lambda g: {'x': 99}, lambda g: {'x': 1}, lambda g: {'x': 5}] * 3)\n"
)


def test_each_statement_runs_once_and_hands_its_value_on(nb_runner):
    (nb_runner.work_dir / "rerun_helpers.py").write_text(HELPERS, encoding="utf-8")
    calls = (nb_runner.work_dir / "calls.log").as_posix()
    nb_runner.create_notebook(
        [
            "import cash\n%cash_on",
            "import os\nname = 'make'\nfrom rerun_helpers import Q, NSS",
            f"def make(x):\n    with open({calls!r}, 'a') as f:\n        f.write('x')\n    return x + 1",
            "# @cash:no-cache\nt = globals()[name](3)",
            "u = t + 0",
            "NS = next(NSS)(globals())",
            "v = NS['x'] + 0",
            "t2 = Q.popleft()",
            "w = t2 + 1",
        ]
    )

    def run_all():
        nb_runner.run_all()
        return (
            len((nb_runner.work_dir / "calls.log").read_text(encoding="utf-8")),
            nb_runner.peek("u"),
            nb_runner.peek("v"),
            nb_runner.peek("w"),
        )

    nb_runner.start_kernel()
    assert run_all() == (1, "4", "1", "1")
    assert run_all() == (2, "4", "99", "2")
    assert run_all() == (3, "4", "1", "3")

    nb_runner.restart()
    assert run_all() == (4, "4", "1", "1")
    assert run_all() == (5, "4", "99", "2")
