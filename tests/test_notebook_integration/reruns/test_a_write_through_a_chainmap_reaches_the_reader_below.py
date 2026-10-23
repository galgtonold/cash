"""A write through a ``ChainMap`` over a dict a frame's column holds reaches
the reader below.

Each notebook edits ``Z = 1`` to ``Z = 2`` after the first Run All. The
statement after it applies ``upd`` to the dicts of ``df['o']``, and ``upd``
writes ``Z`` into each through a ``ChainMap`` made over it (``update``,
``__setitem__``, ``maps[0]``, the parents of ``new_child()``). The
``ChainMap`` was read as a library's result, so the statement wrote
nothing, and the slow reader below printed the value from before the edit.
A plain ``d['z'] = Z`` is the control.

Each is checked over Run All x3 (edited after the first), a restart, and Run
All x2.
"""

from __future__ import annotations

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(600)]

HEAD = "import cash\n%cash_on\nimport time, collections\nimport numpy as np, pandas as pd"

OBJECTS = (
    "class O:\n"
    "    def __init__(self, i):\n"
    "        self.i = i; self.z = 0\n"
    "    def absorb(self, o):\n"
    "        o.z = Z\n"
    "    def __repr__(self):\n"
    "        return f'O({self.i},{self.z})'\n"
)
FRAME = "df = pd.DataFrame({'o': [{'i': i} for i in range(10)], 'p': [O(i) for i in range(10)]})"
READER = "g = (time.sleep(0.3), str(df['o'][3]) + repr(df['p'][3]))[1]\nprint('R', g)"

#: The writer's ``upd``, applied to a column: ``p`` for the objects, ``o``
#: for the dicts.
WRITERS = {
    "plain_setitem": ("o", "def upd(d):\n    d['z'] = Z"),
    "chainmap_update": ("o", "def upd(d):\n    collections.ChainMap(d).update(z=Z)"),
    "chainmap_setitem": ("o", "def upd(d):\n    cm = collections.ChainMap(d)\n    cm.__setitem__('z', Z)"),
    "chainmap_first_map": ("o", "def upd(d):\n    collections.ChainMap(d).maps[0].update(z=Z)"),
    "chainmap_new_child": ("o", "def upd(d):\n    cm = collections.ChainMap(d).new_child()\n    cm.parents['z'] = Z"),
}


def _printed(nb_runner, cell):
    return [line for line in nb_runner.get_output(cell).splitlines() if line.startswith("R ")]


def _rounds(nb_runner, cells, edit_cell, edited):
    """What the last cell printed after Run All, edited, Run All x2, a
    restart, Run All x2."""
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    seen = []
    for run in range(5):
        if run == 1:
            nb_runner.set_cell_source(edit_cell, edited)
        if run == 3:
            nb_runner.restart()
        nb_runner.run_all()
        seen.append(_printed(nb_runner, len(cells)))
    return seen


def _want(writer, z):
    if WRITERS[writer][0] == "p":
        return [f"R {{'i': 3}}O(3,{z})"]
    return [f"R {{'i': 3, 'z': {z}}}O(3,0)"]


@pytest.mark.parametrize("writer", list(WRITERS))
def test_a_write_into_a_column_reaches_the_reader(nb_runner, writer):
    column, upd = WRITERS[writer]
    cells = [
        HEAD,
        OBJECTS + upd + "\n" + FRAME,
        "Z = 1",
        f"_ = (time.sleep(0.3), df[{column!r}].apply(upd))[1]",
        READER,
    ]
    # As plain Python: the frame is made afresh each run, and the writer
    # writes the Z of that run.
    assert _rounds(nb_runner, cells, 3, "Z = 2") == [_want(writer, 1)] + [_want(writer, 2)] * 4
