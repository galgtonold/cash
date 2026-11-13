"""A value built from the ``id()`` of a string the notebook holds is built
again after a restart.

``key = id(holder['a']['b'][0])``, the string built at run time: after a
restart the notebook's string is a new object at a new address. The look for
the object an ``id()`` names never counted strings among what the notebook
holds, so the number was taken for no object's id, ``key`` was stored and
served after the restart with the old process's address, and
``key == id(holder['a']['b'][0])`` was False where a plain Run All gives
True. A list in the same place (the control) was always found.
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
    "target",
    ["'x' * 40 + str(len('abc'))", "1.5 * len('abc')", "[len('abc')]"],
    ids=["a str", "a float", "a list"],
)
@pytest.mark.parametrize(
    ("hold", "read"),
    [
        ("key = slow(id(holder['a']['b'][0]))", "key == id(holder['a']['b'][0])"),
        ("reg = slow({id(holder['a']['b'][0]): 'seen'})", "reg.get(id(holder['a']['b'][0])) == 'seen'"),
    ],
    ids=["bare", "an id key"],
)
def test_the_id_matches_the_held_object_after_a_restart(nb_runner, target, hold, read):
    cells = [
        SETUP,
        SLOW + "holder = slow({'a': {'b': [" + target + "]}})",
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
    if "list" not in target:
        # Not vacuous: the statement before it was served from disk.
        assert "CACHED: holder" in nb_runner.get_raw_output(2), nb_runner.get_raw_output(2)
