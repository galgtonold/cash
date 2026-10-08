"""Records a cell builds come back from the RAM tier as they were built.

The tier keeps big JSON-like data as bytes it reads back in one pass, where
it copied it a container at a time on the store and on every hit. A hit must
still hand back the records as the cell built them: every type, sign and
key order as it was, one list that every record holds still one list, and
none of what a later cell changes in them -- over Run Alls and a restart.
These pass before the change too: they guard it.
"""

import pytest

from tests._nbharness.badge import shows_cached
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.timeout(300)]

N = 20_000

CELLS = [
    "%cash_badge print",
    "import time",
    (
        "records = [{'id': i, 'tags': common, 'meta': {'k': i, 'even': i % 2 == 0, 'neg': -0.0, 'z': 1j, "
        "'raw': b'z', 'big': 2 ** 70}} "
        f"for common in [['x']] for i in range({N}) if i or not time.sleep({ABOVE_PERSISTENCE_FLOOR_S})]"
    ),
    "records[5]['tags'].append('y')\nrecords[7]['meta']['k'] = 'changed'",
    (
        "r = (records[0]['tags'], records[0]['tags'] is records[-1]['tags'], records[7]['meta'], "
        "type(records[2]['meta']['even']).__name__, repr(records[1]['meta']['neg']), len(records))"
    ),
]

EXPECTED = repr(
    (
        ["x", "y"],
        True,
        {"k": "changed", "even": False, "neg": -0.0, "z": 1j, "raw": b"z", "big": 2**70},
        "bool",
        "-0.0",
        N,
    )
)


@pytest.mark.parametrize("imports", ["import time", "import time\nimport numpy as np"], ids=["plain", "numpy-loaded"])
def test_records_restored_from_the_cache_are_the_records_the_cell_built(nb_runner, imports):
    """With numpy loaded the entry also holds numpy's RNG state, and the
    records are kept apart from it."""
    nb_runner.create_notebook([CELLS[0], imports, *CELLS[2:]])
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("r") == EXPECTED, "first Run All"
    for label in ("second", "third"):
        nb_runner.run_all()
        raw = nb_runner.get_raw_output(3)
        assert shows_cached(raw), f"{label} Run All: the records cell is a hit\n{raw}"
        assert nb_runner.peek("r") == EXPECTED, f"{label} Run All"
    nb_runner.restart()
    nb_runner.run_all()
    assert nb_runner.peek("r") == EXPECTED, "after a restart"
