"""joblib's process workers can run a cached function defined in the script.

joblib's default backend (loky) pickles a function from the running script BY
VALUE, and a cached function's closure holds the Cash instance: the call
failed with "Could not pickle the task to send it to the workers". The
function is now pickled by name when the script's work is behind a
``__main__`` guard, so a worker imports the script and decorates it as the
parent did -- same keys, shared entries. Without the guard, the failure says
what to do instead of reporting a lock.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("joblib")

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]

_BODY = '''
import sys, time
import cash
from joblib import Parallel, delayed

@cash.cache(assume_safe=True)
def work(x):
    print("RUN", x, file=sys.stderr)
    time.sleep(0.15)                 # past the persistence floor
    return x * x
'''


def _run(tmp_path, source):
    script = tmp_path / "model.py"
    script.write_text(textwrap.dedent(source), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env.update(CASH_CACHE_DIR=str(tmp_path / ".cash"), PYTHONDONTWRITEBYTECODE="1")
    return subprocess.run([sys.executable, str(script)], cwd=str(tmp_path), env=env,
                          capture_output=True, text=True, timeout=240)


def test_workers_run_a_script_function_and_share_entries_with_the_parent(tmp_path):
    source = _BODY + '''
if __name__ == "__main__":
    print(Parallel(n_jobs=2)(delayed(work)(i) for i in range(4)))
    print("PARENT", file=sys.stderr)
    print(work(3))
'''
    first = _run(tmp_path, source)
    assert first.returncode == 0, first.stderr[-3000:]
    assert first.stdout.split() == ["[0,", "1,", "4,", "9]", "9"], first.stdout
    assert "RUN" not in first.stderr.split("PARENT", 1)[1], \
        "the parent recomputed what a worker had stored"
    again = _run(tmp_path, source)
    assert again.returncode == 0, again.stderr[-3000:]
    assert "RUN" not in again.stderr, "a second run recomputed"


def test_without_a_main_guard_the_failure_says_what_to_do(tmp_path):
    source = _BODY + '''
print(Parallel(n_jobs=2)(delayed(work)(i) for i in range(4)))
'''
    run = _run(tmp_path, source)
    assert run.returncode != 0
    assert 'if __name__ == "__main__":' in run.stderr, run.stderr[-3000:]
