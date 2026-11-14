"""An edited change through a list a module holds reaches the cell below.

``holder.items.append(sessions[1])`` puts a session into another module's
list, and a cell below changes it there:
``holder.items[0]['actions'][-1]['at'] = 2000``. Edited to ``7000`` and run
again, the cell's statement was taken for one edited out of it, so
``holder`` was rebuilt, and ``sessions`` with it, as sharing its object:
``sessions`` came back from the cache as new objects, while the module, the
one ``sys.modules`` holds, kept the old session and the re-run append added
the new one behind it. The change landed on the old session, and the loop
below summed the parse (45) where plain Python and Restart + Run All give
7039. With a closure holding ``sessions[0]`` beside it, only the closure's
change reached the loop.

What shares an object a statement above put into a module is kept live, as
plain Python keeps it. A closure alone is rebuilt with ``sessions``, as
before.
"""

from __future__ import annotations

import sys
import types

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(240)]

PARSE = (
    "import time\n"
    "def parse():\n"
    "    time.sleep(0.3)\n"
    "    return [{'id': i, 'actions': [{'at': j * (i + 1)} for j in range(4)]} for i in range(5)]\n"
    "sessions = parse()"
)
LOOP = (
    "time_intervals = []\n"
    "for s in sessions:\n"
    "    for i in range(len(s['actions']) - 1):\n"
    "        time_intervals.append(s['actions'][i + 1]['at'] - s['actions'][i]['at'])\n"
    "total = sum(time_intervals)"
)
CLOSURE = (
    "def make(s):\n    def put(v):\n        s['actions'][-1]['at'] = v\n    return put\nput_last = make(sessions[0])"
)
MODULE = "import holder\nholder.items.append(sessions[1])"
THROUGH_CLOSURE = "put_last({c})"
THROUGH_MODULE = "holder.items[0]['actions'][-1]['at'] = {m}"

HOLDERS = {
    "both": ([CLOSURE, MODULE], [THROUGH_CLOSURE, THROUGH_MODULE]),
    "swapped": ([MODULE, CLOSURE], [THROUGH_MODULE, THROUGH_CLOSURE]),
    "closure": ([CLOSURE], [THROUGH_CLOSURE]),
    "module": ([MODULE], [THROUGH_MODULE]),
}


def _plain(runs: list[list[str]]) -> int:
    """``below`` after running each list of cells in turn in one namespace,
    with a fresh ``holder`` module: plain Python, without cash."""
    ns: dict = {}
    saved = sys.modules.get("holder")
    sys.modules["holder"] = types.ModuleType("holder")
    sys.modules["holder"].items = []
    try:
        for cells in runs:
            for cell in cells:
                exec(cell.replace("time.sleep(0.3)", "pass"), ns)
    finally:
        if saved is None:
            del sys.modules["holder"]
        else:
            sys.modules["holder"] = saved
    return ns["below"]


@pytest.mark.parametrize("holders", list(HOLDERS))
def test_the_cell_below_sees_the_edited_change(nb_runner, holders):
    hold, through = HOLDERS[holders]
    (nb_runner.work_dir / "holder.py").write_text("items = []\n", encoding="utf-8")

    def change(c: int, m: int) -> str:
        return "\n".join(t.format(c=c, m=m) for t in through)

    cells = [PARSE, "\n".join(hold), LOOP, change(1000, 2000), LOOP + "\nbelow = total"]
    nb_runner.create_notebook(["import cash\n%cash_on", *cells])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("below") == str(_plain([cells]))

    edited = [*cells[:3], change(5000, 7000), cells[4]]
    nb_runner.set_cell_source(5, edited[3])
    nb_runner.run_cells([5, 6])
    # Plain Python and Restart + Run All agree here.
    assert nb_runner.peek("below") == str(_plain([cells, edited[3:]])) == str(_plain([edited]))

    nb_runner.restart()
    nb_runner.run_all()
    assert nb_runner.peek("below") == str(_plain([edited]))
