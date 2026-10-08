"""Records as a parser returns them, built in one cell, and a later cell
that reaches into them.

The cell building the records is checked for other holders and for
closures with state before it is stored; both checks read the records a
level at a time. A name taken from inside the records later must still see
the changes made through it, as in plain Python, over Run Alls and a
restart -- and the first run must not cost many times the build.
"""

import time

import pytest

pytestmark = [pytest.mark.timeout(600)]

BUILD = "[{{'id': i, 'tags': ['a', 'b'], 'meta': {{'k': i}}}} for i in range({n})]"


def _cells(n):
    return [
        f"records = {BUILD.format(n=n)}",
        "first = records[0]",
        "first['meta']['k'] = -1",
        "tags = records[1]['tags']\ntags.append('c')",
        "r = (records[0]['meta']['k'], first is records[0], records[1]['tags'], len(records))",
    ]


def test_changes_through_a_name_from_inside_the_records_are_kept(nb_runner):
    nb_runner.create_notebook(_cells(20_000))
    nb_runner.start_kernel()
    expected = "(-1, True, ['a', 'b', 'c'], 20000)"
    for label in ("first", "second"):
        nb_runner.run_all()
        assert nb_runner.peek("r") == expected, f"{label} Run All"
    nb_runner.restart()
    nb_runner.run_all()
    assert nb_runner.peek("r") == expected, "after a restart"


@pytest.mark.perf
def test_the_first_run_of_a_million_records_is_a_small_multiple_of_building_them(nb_runner):
    """Before: about 5x a build cash does not store, the checks walking one
    container at a time; about 2x since."""
    n = 1_000_000
    nb_runner.create_notebook([f"# @cash:no-cache\nplain = {BUILD.format(n=n)}\ndel plain", f"records = {BUILD.format(n=n)}"])
    nb_runner.start_kernel()
    spent = []
    for cell in (1, 2):
        started = time.perf_counter()
        nb_runner.run_cell(cell)
        spent.append(time.perf_counter() - started)
    assert spent[1] < 3.5 * spent[0], [round(s, 2) for s in spent]
