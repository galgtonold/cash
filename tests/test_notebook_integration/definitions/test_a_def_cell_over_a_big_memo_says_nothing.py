"""A cell that only defines functions says nothing of a helper memo it can
reach, however many answers the memo holds: a ``def`` binds a function and
calls nothing, so the memo decides nothing about it. A statement that
fills such a memo says it filled it, not that it read it.

``rebuild`` names ``cubes.cube.cache_clear``. Run again once
``rebuild(5000)`` left 5,000 answers in ``cube``, more than cash checks,
the cell defining ``rebuild`` was not stored ("It read the helper memo
cubes.cube ...", with ``NOTEBOOK-MEMO-UNREAD``) while the ``rebuild(5000)``
cell below it is served.
"""

from __future__ import annotations

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]

CUBES = "import functools\n@functools.cache\ndef cube(n):\n    return n ** 3\n"
CELLS = [
    "%cash_badge print\nimport time, cubes",
    "def rebuild(k):\n    cubes.cube.cache_clear()\n    time.sleep(0.2)\n    return sum(cubes.cube(i) for i in range(k))",
    "total = rebuild(5000)",
]


def test_the_def_cell_says_nothing_of_the_memo_the_served_call_fills(nb_runner):
    (nb_runner.work_dir / "cubes.py").write_text(CUBES, encoding="utf-8")
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    # The memo was empty: the call fills it past what cash checks.
    first = nb_runner.get_output(3)
    assert "It filled the helper memo cubes.cube, which holds more answers than cash checks" in first, first
    # ``cube`` holds 5,000 answers and ``rebuild`` empties it first: stored,
    # then served.
    nb_runner.run_cell(3)
    nb_runner.run_cell(3)
    assert shows_cached(nb_runner.get_output(3)), nb_runner.get_output(3)
    assert nb_runner.peek("total") == repr(sum(i**3 for i in range(5000)))
    assert nb_runner.peek("cubes.cube.cache_info().currsize") == "5000"
    nb_runner.run_cell(2)
    defined = nb_runner.get_output(2)
    assert "helper memo" not in defined, defined
    assert "NOT CACHED" not in defined, defined
