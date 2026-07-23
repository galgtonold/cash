"""A helper referenced by name but reached through a value must be tracked.

`@cash.cache` walked helper source only from CALL positions, so a helper
reached through a value -- assigned to a local then called, or passed as an
argument -- was never hashed and an edit to it silently served a stale result
(CAS-236). The helper name is still statically visible (it is a read name), so
this is trackable; only genuinely dynamic dispatch (`getattr(m, s)()`) is not.

Cross-process, because the stale-serve only shows up when a second process
rebuilds the key and matches the persisted entry. `time.sleep(0.3)` clears the
persistence floor; without it nothing persists and the test is vacuous.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytestmark = pytest.mark.slow


def _run(tmp_path):
    cp = subprocess.run(
        [sys.executable, "main.py"], cwd=str(tmp_path),
        capture_output=True, text=True,
    )
    assert cp.returncode == 0, f"script failed:\n{cp.stdout}\n{cp.stderr}"
    return cp.stdout.strip()


MAIN = '''\
import warnings; warnings.simplefilter("ignore")
import time
import cash

def helper(x):
    return x + {delta}

def _apply(fn, x):
    return fn(x)

RAN = [0]

@cash.cache
def via_variable(x):
    RAN[0] += 1
    time.sleep(0.3)
    fn = helper          # reached through a value, not called by name
    return fn(x)

@cash.cache
def via_argument(x):
    RAN[0] += 1
    time.sleep(0.3)
    return _apply(helper, x)   # helper passed as an argument

print("V", via_variable(5), "A", via_argument(5), "RAN", RAN[0])
'''


def _write(tmp_path, delta):
    (tmp_path / "main.py").write_text(MAIN.format(delta=delta), encoding="utf-8")


def _vals(out):
    parts = out.split()
    return int(parts[1]), int(parts[3])  # via_variable, via_argument


def test_variable_indirection_and_passed_arg_invalidate(tmp_path):
    _write(tmp_path, "1")
    v1, a1 = _vals(_run(tmp_path))
    assert (v1, a1) == (6, 6)

    # Edit the helper's body. Both indirection forms must recompute.
    _write(tmp_path, "100")
    out2 = _run(tmp_path)
    v2, a2 = _vals(out2)
    assert v2 == 105, f"variable-indirection served a stale value: {v2}"
    assert a2 == 105, f"passed-as-argument served a stale value: {a2}"
