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
    $env:CASH_WHEEL_GATE_ARGS = "--reuse-venv --wheel dist/cash_lib-0.5.0b1-py3-none-any.whl"
    pytest -m wheel_gate tests/test_wheel_gate

The harness exits 0 iff the observed RED/GREEN matrix matches the recorded
baseline: S1/S2 RED (CAS-202 restart-retrain + CAS-196 to_csv re-fire still
reproduce) and S3/S4 GREEN (single-cell sklearn + plain int @cash.cache survive
a restart). This test asserts that exit code, so:

  * a green invariant regressing to RED (S3/S4) fails CI, and
  * a known-open bug getting FIXED (S1/S2 flip to GREEN) also fails CI -- a
    prompt to update the baseline and close the issue.
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
        "wheel-gate baseline mismatch: an invariant regressed (S3/S4 went RED) or "
        "a known-open bug was fixed (S1/S2 went GREEN). See harness output above."
    )
