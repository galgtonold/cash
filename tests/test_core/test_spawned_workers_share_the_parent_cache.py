"""A spawned pool worker keys a script's function under the script's name.

Under the spawn start method (the default on Windows and macOS) a worker
re-imports the script as ``__mp_main__``, not ``__main__``, and only
``__main__`` was resolved to the script's file name. So the workers keyed
``work`` as ``__mp_main__.work`` and the parent as ``model.work``: the
workers shared entries with each other and never with the process that
started them, and ``cash inspect`` listed one function twice.
"""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]


def test_the_parent_hits_what_a_spawned_worker_stored(tmp_path):
    script = tmp_path / "model.py"
    script.write_text(textwrap.dedent('''
        import multiprocessing, sys, time
        import cash

        @cash.cache(assume_safe=True)
        def work(x):
            print("RUN", x, file=sys.stderr)
            time.sleep(0.15)                 # past the persistence floor
            return x * x

        if __name__ == "__main__":
            with multiprocessing.get_context("spawn").Pool(2) as pool:
                print(sorted(pool.map(work, range(4))))
            print("PARENT", file=sys.stderr)
            print(work(3))
    '''), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env.update(CASH_CACHE_DIR=str(tmp_path / ".cash"), PYTHONDONTWRITEBYTECODE="1")
    run = subprocess.run([sys.executable, str(script)], cwd=str(tmp_path), env=env,
                         capture_output=True, text=True, timeout=240)
    assert run.returncode == 0, run.stderr
    assert run.stdout.split() == ["[0,", "1,", "4,", "9]", "9"], run.stdout
    after_pool = run.stderr.split("PARENT", 1)[1]
    assert "RUN" not in after_pool, "the parent recomputed what a worker had stored"

    listing = subprocess.run([sys.executable, "-m", "cash", "inspect", str(tmp_path / ".cash")],
                             env=env, capture_output=True, text=True, timeout=120).stdout
    assert "model.work" in listing and "__mp_main__" not in listing, listing
