"""CI shim for the CAS-190 wheel-gate harness (``scripts/wheel_gate.py``).

This test is SKIPPED by default. The harness it wraps builds a wheel, provisions
a FRESH venv on a short path, and drives a REAL Jupyter server + kernel restarts
-- minutes of work, deliberately kept out of the fast suite (which is
structurally blind to exactly the restart / wheel-venv bugs this catches; that
blindness is the CAS-190 finding).

Run it explicitly::

    # Windows PowerShell
    $env:CASH_WHEEL_GATE = "1"; pytest -m wheel_gate tests/test_wheel_gate

    # reuse an already-provisioned venv to iterate fast
    $env:CASH_WHEEL_GATE = "1"
    $env:CASH_WHEEL_GATE_ARGS = "--reuse-venv --wheel dist/cash_lib-0.1.0-py3-none-any.whl"
    pytest -m wheel_gate tests/test_wheel_gate

The harness exits 0 iff the observed RED/GREEN matrix matches the baseline
recorded in ``scripts/wheel_gate.py``. As of 2026-07-20 that baseline is
**all six scenarios GREEN** (S1-S6), confirmed on the ``0.1.0`` wheel: the bugs
S1/S2/S5 were written to catch (CAS-202 restart-retrain, CAS-196 to_csv
re-fire, CAS-200/193 unrelated-cell plot re-fire) are all fixed, and S3/S4/S6
are controls.

This test asserts that exit code, so a scenario flipping either way fails CI:

  * an invariant regressing to RED means a shipped fix broke, and
  * a scenario unexpectedly changing state means the baseline is stale and the
    matrix in ``scripts/wheel_gate.py`` needs updating alongside the issue.

The direction matters less than the mismatch -- the point is that this file and
the harness's baseline must never disagree about what is currently true.
"""
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "wheel_gate.py"

pytestmark = pytest.mark.wheel_gate


@pytest.mark.timeout(1800)
def test_wheel_gate_matrix():
    if os.environ.get("CASH_WHEEL_GATE") != "1":
        pytest.skip("opt-in: set CASH_WHEEL_GATE=1 to run the slow wheel-venv gate")
    assert SCRIPT.exists(), f"harness not found: {SCRIPT}"
    args = shlex.split(os.environ.get("CASH_WHEEL_GATE_ARGS", ""))
    cp = subprocess.run([sys.executable, str(SCRIPT), *args], cwd=str(REPO), text=True)
    assert cp.returncode == 0, (
        "wheel-gate baseline mismatch: an observed scenario no longer matches the "
        "matrix recorded in scripts/wheel_gate.py -- either a shipped fix "
        "regressed to RED, or a scenario changed state and the baseline is stale. "
        "See harness output above for which one."
    )
