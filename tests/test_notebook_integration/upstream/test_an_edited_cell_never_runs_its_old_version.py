"""A cell edited and run again runs only its new text.

``data`` is rebuilt above the cell (its producer was edited), and the cell's
loop ``for uid, actions in data:`` held objects of the old ``data``: every
name the old loop bound was marked for rebuilding with it. No cell binds them
but the one about to run, so cash took the statement that last bound them,
the cell's own previous loop, for an unsaved edit and ran it before the new
text: the old prints showed above the new ones, and whatever the old loop
did happened twice. A plain run of the notebook as it stands runs the new
loop once.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]

DATA = "data = [({uid!r}, [('a', 1), ('a', 2), ('a', 3)]), ('u2', [('b', 1), ('c', 2), ('c', 3)])]"
LOOP = (
    "total = len(data)\n"
    "k = 0\n"
    "for uid, actions in data:\n"
    "    for i in range(len(actions) - 2):\n"
    "        a_name = actions[i][0]\n"
    "        if a_name == actions[i+1][0] and a_name == actions[i+2][0]:\n"
    "            k += 1\n"
    "            print({show})\n"
    "            break\n"
    "k / total"
)


def _lines(nb_runner, n):
    return [line for line in nb_runner.get_output(n).splitlines() if line.strip()]


def test_only_the_new_text_runs(nb_runner):
    nb_runner.create_notebook([DATA.format(uid="u1"), LOOP.format(show="'old', uid")])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert _lines(nb_runner, 2) == ["old u1", "0.5"]

    nb_runner.set_cell_source(1, DATA.format(uid="u9"))
    nb_runner.set_cell_source(2, LOOP.format(show="'new', uid"))
    nb_runner.run_cell(2)

    assert _lines(nb_runner, 2) == ["new u9", "0.5"]


def test_each_later_edit_runs_only_itself(nb_runner):
    nb_runner.create_notebook([DATA.format(uid="u1"), LOOP.format(show="'first', uid")])
    nb_runner.start_kernel()
    nb_runner.run_all()

    for round_, show in enumerate(["'second', uid", "'third', uid"], start=2):
        nb_runner.set_cell_source(1, DATA.format(uid=f"u{round_}"))
        nb_runner.set_cell_source(2, LOOP.format(show=show))
        nb_runner.run_cell(2)
        word = show.split("'")[1]
        assert _lines(nb_runner, 2) == [f"{word} u{round_}", "0.5"]
