"""OpenMP threads in the test run wait passively (``tests/conftest.py``).

By default they spin between parallel regions, and with a test worker on
every core a spinning thread holds a core the working one needs: the
HistGradientBoosting fit of ``test_a_boosting_fit_is_stored_without_reading_the_heap``
took 110 s instead of 3 s on a loaded 4-core machine, past its timeout on
its first attempt.
"""

from __future__ import annotations

import os
import subprocess
import sys


def test_the_run_and_what_it_starts_wait_passively():
    assert os.environ.get("OMP_WAIT_POLICY") == "PASSIVE"
    seen = subprocess.run(
        [sys.executable, "-c", "import os; print(os.environ.get('OMP_WAIT_POLICY'))"],
        capture_output=True,
        text=True,
        timeout=60,
        check=True,
    )
    assert seen.stdout.strip() == "PASSIVE"
