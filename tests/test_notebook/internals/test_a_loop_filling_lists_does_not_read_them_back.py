"""A loop run as one unit that fills lists does not read them back.

``user_lst.append(u)`` and three more lists of 1.7M items each: the lineage
update after the loop pickled every item to hash the lists, 7.6 s on top of
the loop's own 5.7 s. When the loop's outcome is a function of its key, the
key names the lists instead. Pins the work, not the behaviour: a refactor may
expect this to fail.
"""

from __future__ import annotations

from tests._cell_driver import run_cash_cell
from tests._work_counts import hashed_bytes

SESSIONS = (
    "from datetime import datetime, timedelta\n"
    "sessions = [[('u%d' % i, j, datetime(2019, 1, 1) + timedelta(seconds=i * 20 + j), 'Action_%d' % (j % 11))"
    " for j in range(20)] for i in range(5000)]"
)
LOOP = (
    "user_lst, num_lst, time_lst, name_lst = [], [], [], []\n"
    "for s in sessions:\n    for (u, j, t, act) in s:\n        user_lst.append(u)\n        num_lst.append(j)\n"
    "        time_lst.append(t)\n        name_lst.append(act)"
)


def _read_the_classes_alive(namespace):
    """What the measured loop reads once a process, read first. A statement
    reading a container too big to look through asks whether a class alive
    has a special method (`special_classes_exist`, which stops at the first
    it finds) and, when one has, reads the special methods of every class
    alive (`unseen_writes._live_writing_classes`) -- in a test process,
    other tests' classes, all of them once the whole suite is collected. The
    first read of a file in a process digests its text (`_kept_block`), which
    the counter takes for hashing: 87 KB to 1 MB, by what ran before in that
    worker and what was collected. Read here, the count is the loop's own."""
    from cash.analysis import handed_callables as hc
    from cash.notebook.statement.unseen_writes import _live_writing_classes

    if hc.special_classes_exist(namespace):
        _live_writing_classes(namespace, hc)


def test_the_lists_are_not_hashed(cash_magics):
    run_cash_cell(cash_magics, SESSIONS)
    _read_the_classes_alive(cash_magics.shell.user_ns)
    with hashed_bytes() as hashed:
        run_cash_cell(cash_magics, LOOP)
    assert len(cash_magics.shell.user_ns["user_lst"]) == 100_000
    assert hashed.bytes < 64 * 1024  # ~4 MB before: every item pickled


def test_a_loop_drawing_random_numbers_still_hashes_them(cash_magics):
    """The control: an outcome that is not a function of the key is read."""
    run_cash_cell(cash_magics, "import random\nrows = list(range(100_000))")
    with hashed_bytes() as hashed:
        run_cash_cell(
            cash_magics, "out = []\nfor r in rows:\n    for _ in range(2):\n        out.append(random.random())"
        )
    assert hashed.bytes >= 1_000_000
