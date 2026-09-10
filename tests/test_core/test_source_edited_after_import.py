"""A helper edited on disk after import is keyed by the code that runs.

CAS-110, round-17 tester r17s3. The ordinary deploy sequence:

    process A: import app              (helper.bump is `x + 1`)
               -- new files land: helper.py now says `x + 100` --
               compute(1) -> 2, stored under the NEW source's key
    process B: compute(1) -> HIT, 2.   The right answer is 101.

A helper's digest is computed lazily, at the first call that needs it, from
the text on disk -- which by then described the new code while the old code
object was the one executing.

Now a helper whose file changed after the process started is compiled from
disk (the whole file, so the compiler sees the module context the import did)
and compared with the loaded code; if they differ, it is keyed by the loaded
bytecode, and KEY-SOURCE-CHANGED says so. The restarted process keys the new
code by its source, a different key, and computes.

Process A is held between import and its first call with a handshake file,
not a sleep, and bytecode caching is off so a stale `.pyc` cannot fake either
outcome (r17s3 hit that trap once).
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap
import time

import pytest

pytestmark = pytest.mark.core

OLD = "def bump(x):\n    return x + 1\n"
NEW = "def bump(x):\n    return x + 100\n"

MAIN = textwrap.dedent('''
    import os, sys, time
    import cash
    from helper import bump

    @cash.cache
    def compute(x):
        print("COMPUTE", file=sys.stderr, flush=True)  # @cash:assume-safe
        time.sleep(0.25)
        return bump(x)

    if __name__ == "__main__":
        if len(sys.argv) > 1 and sys.argv[1] == "wait":
            open("ready", "w").close()
            for _ in range(400):
                if os.path.exists("go"):
                    break
                time.sleep(0.025)
        print(compute(1))
''')


def _project(tmp_path, helper_text):
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / "helper.py").write_text(helper_text, encoding="utf-8")
    (proj / "main.py").write_text(MAIN, encoding="utf-8")
    return proj


def _env(tmp_path):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env["CASH_CACHE_DIR"] = str(tmp_path / "cache")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _edit(path, text):
    """Rewrite *path* and make sure its mtime moves, even on a coarse clock."""
    before = os.stat(path).st_mtime_ns
    path.write_text(text, encoding="utf-8")
    if os.stat(path).st_mtime_ns == before:
        os.utime(path, ns=(before + 1_000_000, before + 1_000_000))


def _run(proj, env, *argv):
    return subprocess.run([sys.executable, "main.py", *argv], cwd=str(proj),
                          capture_output=True, text=True, env=env)


def test_an_edit_between_import_and_first_call_is_not_served_to_the_restart(tmp_path):
    """THE BUG: the restarted process got the old code's answer."""
    proj = _project(tmp_path, OLD)
    env = _env(tmp_path)

    a = subprocess.Popen([sys.executable, "main.py", "wait"], cwd=str(proj),
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    for _ in range(400):
        if (proj / "ready").exists():
            break
        time.sleep(0.025)
    assert (proj / "ready").exists(), "process A never reached its first call"
    _edit(proj / "helper.py", NEW)             # the deploy lands under A
    (proj / "go").write_text("", encoding="utf-8")
    a_out, a_err = a.communicate(timeout=60)

    assert a_out.strip() == "2", "A runs the code it imported"
    assert "KEY-SOURCE-CHANGED" in a_err, "nothing said the file changed under A"

    b = _run(proj, env)                         # the restart
    assert b.stdout.strip() == "101", "the restarted process was served the old code's answer"
    assert "COMPUTE" in b.stderr


def test_an_edit_before_the_process_starts_is_ordinary(tmp_path):
    """Control: the normal restart order computes once, then hits."""
    proj = _project(tmp_path, OLD)
    env = _env(tmp_path)
    _edit(proj / "helper.py", NEW)

    first = _run(proj, env)
    second = _run(proj, env)

    assert first.stdout.strip() == second.stdout.strip() == "101"
    assert "COMPUTE" not in second.stderr, "an ordinary restart stopped hitting"
    assert "KEY-SOURCE-CHANGED" not in first.stderr + second.stderr


def test_an_unedited_helper_is_silent(tmp_path):
    """Control: no edit, no notice, and the second run hits."""
    proj = _project(tmp_path, OLD)
    env = _env(tmp_path)

    first = _run(proj, env)
    second = _run(proj, env)

    assert first.stdout.strip() == second.stdout.strip() == "2"
    assert "COMPUTE" not in second.stderr
    assert "KEY-SOURCE-CHANGED" not in first.stderr + second.stderr
