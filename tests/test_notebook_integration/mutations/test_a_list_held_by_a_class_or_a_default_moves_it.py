"""A list a class attribute or a function's default holds moves the class
or the function when it changes in place.

``Reg.items.append(rows)`` (or ``collect(rows)`` with ``def collect(x=None,
acc=[])``), then ``rows.append(3)``: the class and the function were taken
for values no statement changes, so after an edit to ``rows.append(...)``
the cached ``slow(Reg.items)`` / ``slow(collect())`` was served for the old
list, on Run All and on a run of the reading cell alone.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.mutations, pytest.mark.timeout(300)]

SETUP = "import cash\n%cash_on\nimport time\ndef slow(x):\n    time.sleep(0.3)\n    return repr(x)"
CASES = {
    "class": ("class Reg:\n    items = []", "Reg.items", "Reg.items.append(rows)"),
    "default": (
        "def collect(x=None, acc=[]):\n    if x is not None:\n        acc.append(x)\n    return acc",
        "collect()",
        "collect(rows)",
    ),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_an_edited_change_reaches_the_reader_of_the_holder(nb_runner, case):
    holder, use, put = CASES[case]
    nb_runner.create_notebook([SETUP, holder, "rows = [1, 2]\n" + put, "rows.append(3)", f"r = slow({use})\nprint(r)"])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "[[1, 2, 3]]" in nb_runner.get_output(5), "control: the first run"

    nb_runner.set_cell_source(4, "rows.append(4)")
    nb_runner.run_all()
    assert "[[1, 2, 4]]" in nb_runner.get_output(5), nb_runner.get_output(5)

    nb_runner.set_cell_source(4, "rows.append(5)")
    nb_runner.run_cells([5])
    assert "[[1, 2, 5]]" in nb_runner.get_output(5), nb_runner.get_output(5)
