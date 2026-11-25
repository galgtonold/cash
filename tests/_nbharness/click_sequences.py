"""Short click sequences on small notebooks, each checked against a plain run from the top.

A user's clicks are a sequence: Run All, edit a cell, restart the kernel,
run one cell further down. What cash must show after each run is what the
notebook, as it stands, prints when a plain kernel runs it from the top
(``session_harness.Player`` checks exactly that after every run). The bugs
this finds are those where the kernel that ran a statement knew something
about it that the next kernel, or the check of a cell run out of order, has
to work out again: a list a helper fills, a setting of the process, a
closure's state.

A sequence is a notebook and its clicks::

    clicks("a list a helper fills", [cell, cell, ...], RunAll(), Restart(), run(4))

Cells are numbered from 1 as written here; the harness puts the cell that
turns cash on above them. A cell says what it holds by printing it: the
check compares what each run prints.

``restart_then_each_alone`` builds the heavier family: for every cell, Run
All, restart, run that cell alone.
"""

from __future__ import annotations

from tests._nbharness.session_harness import Edit, Restart, Run, RunAll, Session

__all__ = ["SEQUENCES", "EACH_CELL_ALONE", "clicks", "edit", "restart_then_each_alone", "run"]

SETUP = "import cash\n%cash_on"


def run(cell: int) -> Run:
    """Run cell number *cell* alone."""
    return Run(str(cell))


def edit(cell: int, source: str) -> Edit:
    """Replace the source of cell number *cell*."""
    return Edit(str(cell), source)


def clicks(name: str, cells: list[str], *steps) -> Session:
    """The sequence *steps* on a notebook of *cells*."""
    labelled = (("setup", SETUP), *((str(n), source) for n, source in enumerate(cells, start=1)))
    return Session(name, labelled, (), tuple(steps))


def restart_then_each_alone(name: str, cells: list[str]) -> Session:
    """Run All, then for each cell: restart and run it alone."""
    steps: list = [RunAll()]
    for n in range(1, len(cells) + 1):
        steps += [Restart(), run(n)]
    return clicks(name, cells, *steps)


_LOG = "import time\nLOG = []\ndef rec(v):\n    LOG.append(v)\n    time.sleep(0.3)\n    return v * 2"
_COUNTER = "def make():\n    n = 0\n    def inc():\n        nonlocal n\n        n += 1\n        return n\n    return inc\ninc = make()"

#: The notebooks, by what the cells above the last one change behind its back.
NOTEBOOKS: dict[str, list[str]] = {
    "a list a helper fills": [_LOG, "a = rec(1)", "b = rec(2)", "c = len(LOG)\nprint(c, LOG)"],
    "a helper called again below": [_LOG, "a = rec(1)", "b = rec(2)", "c = rec(3)\nprint(c, LOG)"],
    "a counter under global": [
        "step = 0\ndef train(x):\n    global step\n    step += 1\n    return x * 2",
        "l1 = train(1)",
        "l2 = train(2)",
        "print(step, l1, l2)",
    ],
    "a bound method kept in a variable": ["lst = [1, 2, 3]", "f = lst.append\nf(4)", "print(lst)"],
    "a bound method called in a loop": ["lst = [1]\nf = lst.append", "for x in [2, 3]:\n    f(x)", "print(lst)"],
    "a closure that counts": [_COUNTER, "a = inc()", "b = inc()", "c = inc()\nprint(c)"],
    "a registry a decorator fills": [
        "registry = []\ndef register(f):\n    registry.append(f.__name__)\n    return f",
        "@register\ndef a(): pass",
        "@register\ndef b(): pass",
        "print(registry)",
    ],
    "settings of the process": [
        "import os, time\nfrom decimal import Decimal, getcontext\ngetcontext().prec = 6\nos.environ['CASH_TEST_CLICKS'] = 'p'",
        "time.sleep(0.3)\nprice = Decimal('19.99')\ndel os.environ['CASH_TEST_CLICKS']",
        "print(price / 7, os.getenv('CASH_TEST_CLICKS'))",
    ],
    "a variable read through eval": ["K = 2", 'y = eval("K * 10")', "print(y)"],
}

#: A fixed handful: Run All, restart, run the last cell alone, for each
#: notebook, and a few longer walks.
SEQUENCES: list[Session] = [
    *(clicks(name, cells, RunAll(), Restart(), run(len(cells))) for name, cells in NOTEBOOKS.items()),
    clicks(
        "an argument edited before the restart",
        NOTEBOOKS["a list a helper fills"],
        RunAll(),
        edit(2, "a = rec(5)"),
        Restart(),
        run(4),
    ),
    clicks(
        "an edit above, then a restart",
        NOTEBOOKS["a bound method kept in a variable"],
        RunAll(),
        edit(1, "lst = [7, 8]"),
        run(3),
        Restart(),
        run(3),
    ),
    clicks(
        "a middle cell, then the last, after a restart",
        NOTEBOOKS["a list a helper fills"],
        RunAll(),
        Restart(),
        run(3),
        run(4),
    ),
    clicks(
        "two restarts",
        NOTEBOOKS["a closure that counts"],
        RunAll(),
        Restart(),
        run(4),
        Restart(),
        run(4),
    ),
]

#: The heavier family: every cell alone after a restart.
EACH_CELL_ALONE: list[Session] = [
    restart_then_each_alone(name, NOTEBOOKS[name])
    for name in ("a list a helper fills", "a bound method kept in a variable", "a closure that counts")
]
