"""``w = mk(df)``, a call handing back its argument, is served after a restart.

After a restart ``df`` is restored, a new object, so the statement's
in-place-change fingerprint hashes it before it runs and keeps ``(df,
digest)`` to hand to the call (`DigestHandoff`). The hit then asked whether
anything but ``df``'s name holds the object, counted that tuple, and ran the
statement again every time. It is cash's own reference.
"""

import time

import pytest

from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

pytestmark = [pytest.mark.integration, pytest.mark.timeout(300)]

SLEEP = ABOVE_PERSISTENCE_FLOOR_S * 5
CELLS = [
    "import cash\n%cash_on\nimport time\nimport numpy as np, pandas as pd",
    "df = pd.DataFrame({'x': np.arange(10), 'o': [{'i': i} for i in range(10)]})",
    f"def mk(v):\n    time.sleep({SLEEP})\n    return v",
    "w = mk(df)",
    "print('R', w is df)",
]


def test_the_call_is_served(nb_runner):
    nb_runner.create_notebook(CELLS)
    nb_runner.start_kernel()
    nb_runner.run_all()
    assert "R True" in nb_runner.get_output(5), "control: the first run"

    nb_runner.restart()
    nb_runner.run_cells([1, 2, 3])
    started = time.perf_counter()
    nb_runner.run_cell(4)
    took = time.perf_counter() - started
    nb_runner.run_cell(5)

    assert "R True" in nb_runner.get_output(5), nb_runner.get_output(5)
    assert took < SLEEP, f"ran again: {took:.2f} s"
