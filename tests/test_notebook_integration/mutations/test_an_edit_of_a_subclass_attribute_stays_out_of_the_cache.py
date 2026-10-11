"""A later cell's edit of a table subclass's own attribute stays out of the cache.

pandas copies a subclass's ``_metadata`` attributes (here ``df.info``) by
reference into every copy, so the RAM entry, the restored table and every
later hit held one dict: each unchanged Run All appended to it again. Every
Run All must print what the first, plain one printed.
"""

import pytest

from tests._nbharness.badge import shows_cached
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.timeout(300)]

CELLS = [
    "import cash\n%cash_on\n%cash_badge print",
    "import time, pandas as pd, numpy as np",
    "class Tagged(pd.DataFrame):\n    _metadata = ['info']\n    @property\n    def _constructor(self):\n        return Tagged",
    (
        f"def load():\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})\n"
        "    f = Tagged({'a': np.arange(5.0)})\n    f.info = {'steps': []}\n    return f"
    ),
    "df = load()",
    "df.info['steps'].append('cleaned')",
    "r = repr(df.info)",
]

EXPECTED = repr("{'steps': ['cleaned']}")


def test_every_run_all_gets_the_attribute_the_function_built(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert nb_runner.peek("r") == EXPECTED, "first Run All"
    for label in ("second", "third"):
        nb_runner.run_all()
        raw = nb_runner.get_raw_output(5)
        assert shows_cached(raw), f"{label} Run All: the load cell is a hit\n{raw}"
        assert nb_runner.peek("r") == EXPECTED, f"{label} Run All"
