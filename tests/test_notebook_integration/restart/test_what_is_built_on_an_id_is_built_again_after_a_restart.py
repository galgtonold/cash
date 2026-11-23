"""What is built from a value holding an ``id()`` is built again after a
restart.

``keys = [... str(id(p.child))]`` runs every time (it holds an id()), but its
lineage named only its code and inputs, the same after a restart. So
``df = make_frame(keys)`` below was served its call by that lineage, built
from the old process's address, where a plain Run All builds it from the
new one. The same for an id() written into a list an earlier statement
built (``keys[3] = id(p.child)``).
"""

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.restore, pytest.mark.timeout(240)]

SETUP = (
    "import cash\n%load_ext cash\n%cash_badge print\n"
    "cash.configure(call_cost_floor_seconds=0.0, min_execution_time_to_cache_seconds=0.0)\n%cash_on"
)

#: Slow enough to be kept on disk; a list of 20,000 texts is keyed by its
#: lineage, not hashed per call.
HELPER = (
    "import time\nimport pandas as pd\n"
    "def make_frame(k):\n    time.sleep(0.3)\n    return pd.DataFrame({'key': pd.array(k, dtype='string')})\n"
    "class P:\n    pass\np = P()\np.child = P()\n"
)
BIG = "[f'k{i}' for i in range(20_000)]"


@pytest.mark.parametrize(
    ("hold", "read"),
    [
        (f"keys = {BIG} + [str(id(p.child))]", "df['key'].iloc[-1] == str(id(p.child))"),
        ("keys = [str(id(p.child))] * 20_000", "df['key'].iloc[-1] == str(id(p.child))"),
        (f"keys = {BIG}\nkeys[-1] = str(id(p.child))", "df['key'].iloc[-1] == str(id(p.child))"),
        (f"keys = {BIG}\nkeys[3] = id(p.child)", "keys[3] == id(p.child) and df['key'].iloc[3] == str(id(p.child))"),
    ],
    ids=["in-a-list", "the-exact-text", "written-in-place", "an-int-written-in-place"],
)
def test_a_frame_built_on_an_id_matches_the_object_after_a_restart(nb_runner, hold, read):
    cells = [SETUP, HELPER, hold, "df = make_frame(keys)", f"# @cash:no-cache\nprint('R', {read})"]
    nb_runner.create_notebook(cells)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R True" in nb_runner.get_output(5), nb_runner.get_raw_output(5)

    nb_runner.restart()
    nb_runner.run_all()

    assert "R True" in nb_runner.get_output(5), [nb_runner.get_raw_output(i) for i in (3, 4, 5)]
