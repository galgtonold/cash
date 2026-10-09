"""A loop run as one unit that fills lists names them by what went in.

``user_lst.append(u)`` and three more lists of 1.7M items: the lineage update
after the loop read every item back (7.6 s on top of the loop's 5.7 s). When
the loop's outcome is a function of its key, the key names the lists; a cell
reading them still follows an upstream edit.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SETUP = "import cash\n%load_ext cash\n%cash_badge print\n%cash_on"
SESSIONS = "sessions = [[('u%d' % i, j * {k}) for j in range(20)] for i in range(3000)]"
LOOP = (
    "user_lst, num_lst = [], []\n"
    "for s in sessions:\n    for (u, j) in s:\n        user_lst.append(u)\n        num_lst.append(j)"
)
COUNT = (
    "import cash.notebook.control_structures.helpers as h\n"
    "if not hasattr(h, 'real_update'):\n"
    "    h.real_update = h.update_mutated_variable_lineages\n"
    "    h.read_back = []\n"
    "    def counting(shell, sp, names, *a, h=h, **k):\n"
    "        if not k.get('unit_digest'):\n"
    "            h.read_back.extend(sorted(names))\n"
    "        return h.real_update(shell, sp, names, *a, **k)\n"
    "    h.update_mutated_variable_lineages = counting\n"
)
READ_BACK = "__import__('cash.notebook.control_structures.helpers').notebook.control_structures.helpers.read_back"


def test_the_lists_are_named_by_the_key_and_follow_an_edit(nb_runner):
    nb_runner.create_notebook([SETUP, SESSIONS.format(k=1), LOOP, "print('SUM', sum(num_lst), len(user_lst))"])
    nb_runner.start_kernel()
    nb_runner.run_cells([1, 2])
    nb_runner.peek(f"exec({COUNT!r})")
    nb_runner.run_cells([3, 4])
    assert "SUM 570000 60000" in nb_runner.get_output(4)
    assert nb_runner.peek(f"len({READ_BACK})") == "0"

    nb_runner.set_cell_source(2, SESSIONS.format(k=2))
    nb_runner.run_cells([2, 3, 4])
    assert "SUM 1140000 60000" in nb_runner.get_output(4)
