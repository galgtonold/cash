"""A value whose text spells an object's ``id()`` inside it is built again
after a restart.

``key = slow('k' + str(id(sales)))``: the identity check read a text only
when it was a number as a whole, so ``key`` was stored and served after the
restart with the old process's address, and ``key == 'k' + str(id(sales))``
was False where a plain Run All gives True.
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.restore, pytest.mark.timeout(240)]

SETUP = (
    "import cash\n%load_ext cash\n%cash_badge print\n"
    "cash.configure(call_cost_floor_seconds=0.0, min_execution_time_to_cache_seconds=0.0)\n%cash_on"
)

#: Worth keeping on disk, so a restart can serve it.
SLOW = "import time\ndef slow(v):\n    time.sleep(0.3)\n    return v\n"


@pytest.mark.parametrize(
    ("hold", "read"),
    [
        ("key = slow('k' + str(id(sales)))", "key == 'k' + str(id(sales))"),
        ("key = slow(f'run-{hex(id(sales))}')", "key == f'run-{hex(id(sales))}'"),
    ],
    ids=["decimal", "hex"],
)
def test_the_spelled_id_matches_the_held_object_after_a_restart(nb_runner, hold, read):
    cells = [
        SETUP,
        SLOW + "sales = slow([len('abc')])",
        hold,
        f"# @cash:no-cache\nprint('R', {read})",
    ]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R True" in nb_runner.get_output(4), nb_runner.get_raw_output(4)

    nb_runner.restart()
    nb_runner.run_all()

    assert "R True" in nb_runner.get_output(4), (nb_runner.get_raw_output(3), nb_runner.get_raw_output(4))
