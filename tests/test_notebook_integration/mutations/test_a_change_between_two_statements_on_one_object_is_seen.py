"""A change made to an object between two statements that change it in place
is seen, as plain Python sees it, after a restart too.

A notebook keeps one big object and runs many tiny statements on it
(``box.flag = 2``). Between two of them, code other than the next statement
can change the object: its display (IPython calls ``__repr__``), a hook
IPython runs after each cell (here a list's own ``append``, which runs no
Python code of its own), a timer thread, another cell. A digest of the
object taken after one statement says nothing of it at the start of the
next, and a reader below must see every one of those changes. The oracle is
the same notebook in a kernel without cash.
"""

import pytest

pytestmark = [pytest.mark.timeout(240)]

_CELLS = [
    (
        "import threading, time\n"
        "class Box:\n"
        "    def __init__(self):\n"
        "        self.items = list(range(5000))\n"
        "        self.log = []\n"
        "        self.flag = 0\n"
        "    def __repr__(self):\n"
        "        self.log.append('shown')\n"
        "        return 'Box'\n"
        "box = Box()"
    ),
    "box.flag = 1",
    "box",  # the display changes it
    "box.flag = 2",
    "get_ipython().events.register('post_run_cell', box.log.append)",
    "box.flag = 3",  # the hook appends after each cell from here
    "box.log.append('cell')",  # another cell
    "box.flag = 4",
    "threading.Timer(0.0, box.log.append, args=('timer',)).start()",
    "time.sleep(0.5)\nbox.flag = 5",
    "get_ipython().events.unregister('post_run_cell', box.log.append)",
    "kinds = [x if isinstance(x, str) else type(x).__name__ for x in box.log]\nn = len(box.items) + box.flag",
    "print('SEEN', n, kinds)",
]


def _seen(nb_runner) -> str:
    out = nb_runner.get_output(len(_CELLS) + 1)
    return next(line for line in out.splitlines() if line.startswith("SEEN"))


def test_each_change_between_statements_is_seen_as_plain_python_sees_it(nb_runner):
    nb_runner.create_notebook(["pass", *_CELLS])
    nb_runner.start_kernel(with_cash=False)
    nb_runner.assert_cash_active(False)
    nb_runner.run_all()
    plain = _seen(nb_runner)
    for kind in ("shown", "ExecutionResult", "cell", "timer"):
        assert kind in plain, plain
    nb_runner.shutdown()

    nb_runner.create_notebook(["import cash\n%cash_on", *_CELLS])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _seen(nb_runner) == plain
    nb_runner.restart()
    nb_runner.run_all()
    assert _seen(nb_runner) == plain
