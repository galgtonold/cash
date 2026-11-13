"""A helper rewritten at the same size with its old mtime put back is a new helper.

``git checkout``, ``cp -p``, ``tar`` and ``rsync -t`` rewrite files without
moving what a quick stat compares: a same-size edit (``+ 10`` -> ``+ 20``) can
keep the size and, restored or inside one tick, the mtime. Python's ``.pyc``
check compares just those (the mtime in whole seconds), so a restarted process
may run the old bytecode; cash must key that run by the code that runs, and
never leave its result under the new text's key for the process that finally
runs the new code.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
import time

import pytest

from cash.decorator import code_trace
from tests._scripts import child_env

pytestmark = pytest.mark.core

OLD = "def bump(x):\n    return x + 10\n"
NEW = "def bump(x):\n    return x + 20\n"

MAIN = textwrap.dedent("""
    import cash
    import cash.loaded_code
    if WINDOWS:  # st_ctime is the creation time there: no change time to go by
        cash.loaded_code._HAS_CHANGE_TIME = False
    from helper import bump

    @cash.cache
    def compute(x):
        return bump(x)

    class Box:
        @cash.cache
        def compute(self, x):
            return bump(x)

    print(compute(1), Box().compute(1))
""")


def _run(proj, env):
    out = subprocess.run([sys.executable, "main.py"], cwd=str(proj), capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stderr
    return out.stdout.split(), out.stderr


@pytest.mark.parametrize("windows", [False, True], ids=["change-time", "no-change-time"])
@pytest.mark.parametrize("shift_ns", [0, 1000], ids=["identical-mtime", "same-second"])
def test_a_same_size_rewrite_with_the_mtime_put_back_is_not_served_to_the_new_code(tmp_path, shift_ns, windows):
    proj = tmp_path / "proj"
    proj.mkdir()
    helper = proj / "helper.py"
    helper.write_text(OLD, encoding="utf-8")
    old = int(time.time() - 100) * 1_000_000_000
    os.utime(helper, ns=(old, old))
    (proj / "main.py").write_text(f"WINDOWS = {windows}\n" + MAIN, encoding="utf-8")
    env = child_env(cache_dir=tmp_path / "cache")
    env.pop("PYTHONDONTWRITEBYTECODE", None)  # the .pyc is the point

    assert _run(proj, env)[0] == ["11", "11"]
    helper.write_text(NEW, encoding="utf-8")
    os.utime(helper, ns=(old, old + shift_ns))

    # Python runs the stale bytecode (its header still matches): the old
    # code's answer, said to be the old code's.
    got, err = _run(proj, env)
    assert got == ["11", "11"]
    assert "KEY-SOURCE-CHANGED" in err and ".pyc" in err

    shutil.rmtree(proj / "__pycache__")
    assert _run(proj, env)[0] == ["21", "21"], "the stale run's result was stored under the new text's key"


def test_a_file_digest_is_not_kept_across_a_same_stat_rewrite(tmp_path):
    """The trace's digest of a module file: a same-size save that puts the
    mtime back, or lands in the tick the digest was read in, is read again."""
    path = tmp_path / "mod.py"
    path.write_text(OLD, encoding="utf-8")
    first = os.stat(path)
    before = code_trace._file_digest(str(path))
    path.write_text(NEW, encoding="utf-8")
    os.utime(path, ns=(first.st_atime_ns, first.st_mtime_ns))
    assert code_trace._file_digest(str(path)) != before
