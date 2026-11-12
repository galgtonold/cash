"""An array a cell writes into is the same array after a cache hit.

``arr[3] = 7`` stored, then ``reg = {id(arr): 'a'}`` added in a cell above
it: a hit bound ``arr`` to the stored copy, so ``reg.get(id(arr))`` gave
None where plain Python gives ``'a'``. The hit now writes the stored state
into the array ``arr`` is bound to, and is still served.
"""

import pytest

from tests._nbharness.badge import shows_cached

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

PIN = "cash.configure(call_cost_floor_seconds=0.0, min_execution_time_to_cache_seconds=0.0)\n"
SETUP = "import cash\n%load_ext cash\n%cash_badge print\n" + PIN + "%cash_on"
MAKE = "import numpy as np\narr = np.zeros(10)"
WRITE = "arr[3] = 7"
CHECK = "print('R', reg.get(id(arr)), float(arr[3]))"
PLAIN = "R a 7.0"


def test_an_id_key_taken_after_the_store_finds_the_array(nb_runner):
    nb_runner.create_notebook([SETUP, MAKE, "reg = {}", WRITE, CHECK])
    nb_runner.start_kernel()
    nb_runner.run_all()
    nb_runner.set_cell_source(3, "reg = {id(arr): 'a'}")
    for _ in range(2):
        nb_runner.run_all()
        assert PLAIN in nb_runner.get_output(5), nb_runner.get_raw_output(5)
    assert shows_cached(nb_runner.get_output(4)), nb_runner.get_raw_output(4)
