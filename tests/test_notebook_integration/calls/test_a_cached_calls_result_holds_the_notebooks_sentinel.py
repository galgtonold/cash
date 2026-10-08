"""A cached call's result holds the notebook's own sentinel on every Run All
and after a restart.

``MISSING = object()`` in a cell, ``res = lookup(keys)`` with the call
cached on its own: the hit's result held a copy of ``MISSING``, so
``v is MISSING`` was False from the second Run All on. Plain Jupyter
prints ``[False, True]`` every time.
"""

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.integration, pytest.mark.timeout(120)]


def test_a_sentinel_in_a_cached_calls_result_is_the_notebooks_own(nb_runner):
    nb_runner.create_notebook(
        [
            "import time\nMISSING = object()\nTABLE = {'a': 1}",
            f"def lookup(keys):\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S * 2})\n"
            "    return [TABLE.get(k, MISSING) for k in keys]",
            "res = lookup(('a', 'b'))",
            "missing = [x is MISSING for x in res]",
        ]
    )
    nb_runner.start_kernel()
    for label in ("Run All 1", "Run All 2"):
        nb_runner.run_all()
        assert nb_runner.peek("missing") == "[False, True]", label
    nb_runner.restart()
    nb_runner.run_all()
    assert nb_runner.peek("missing") == "[False, True]", "after a restart"
