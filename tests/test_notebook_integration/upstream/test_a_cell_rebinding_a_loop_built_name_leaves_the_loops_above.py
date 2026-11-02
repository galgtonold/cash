"""Re-running a cell whose loop rebinds a name a loop above built leaves
the cells above alone.

``actions`` is built by the loop that builds ``data``; the cell below binds
it again as a loop target (``for uid, actions in data:``). On the next run of
that cell the last statement to bind ``actions`` was the cell's own loop,
which cash took for an unsaved edit above it: it stopped trusting every name
a loop built, found ``data`` out of date, and ran its producers again before
the cell. A plain run of the cell leaves the cells above alone.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.upstream]

# Counts the rows the build loop reads, in a file a cache hit does not restore.
HELPER = """
import os
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rows.log")
def note():
    fd = os.open(LOG, os.O_WRONLY | os.O_APPEND | os.O_CREAT)  # @cash:assume-safe
    os.write(fd, b"row\\n")
    os.close(fd)
"""
RAW = "import h\nraw = ['u1,a:1,a:2,a:3', 'u2,b:1,c:2,c:3']"
BUILD = (
    "data = []\n"
    "actions = []\n"
    "for row in raw:\n"
    "    h.note()\n"
    "    parts = row.split(',')\n"
    "    acts = []\n"
    "    for p in parts[1:]:\n"
    "        a, t = p.split(':')\n"
    "        acts.append((a, int(t)))\n"
    "        actions.append(a)\n"
    "    data.append((parts[0], acts))"
)
SORT = "data.sort(key=lambda x: x[-1][0][-1])"
COUNT = (
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


@pytest.mark.parametrize("edit", [False, True], ids=["unchanged", "edited"])
def test_the_cells_above_are_not_run_again(nb_runner, tmp_path, edit):
    (tmp_path / "h.py").write_text(HELPER, encoding="utf-8")
    nb_runner.create_notebook([RAW, BUILD, SORT, COUNT.format(show="'first', uid")])
    nb_runner.start_kernel()
    nb_runner.run_all()
    rows = tmp_path / "rows.log"
    assert rows.read_text().split() == ["row", "row"]
    assert _lines(nb_runner, 4) == ["first u1", "0.5"]

    if edit:
        nb_runner.set_cell_source(4, COUNT.format(show="'second', uid"))
    nb_runner.run_cell(4)

    word = "second" if edit else "first"
    assert _lines(nb_runner, 4) == [f"{word} u1", "0.5"]
    assert nb_runner.peek("[uid for uid, _ in data]") == "['u1', 'u2']"
    assert rows.read_text().split() == ["row", "row"], "the loop building data ran again"
