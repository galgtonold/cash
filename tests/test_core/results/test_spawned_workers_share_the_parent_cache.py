"""A spawned pool worker keys a script's function under the script's name.

Under the spawn start method (the default on Windows and macOS) a worker
re-imports the script as ``__mp_main__``, not ``__main__``, and only
``__main__`` was resolved to the script's file name. So the workers keyed
``work`` as ``__mp_main__.work`` and the parent as ``model.work``: the
workers shared entries with each other and never with the process that
started them, and ``cash inspect`` listed one function twice.
"""

from __future__ import annotations

import textwrap

import pytest

from tests._scripts import run_python

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]


def test_the_parent_hits_what_a_spawned_worker_stored(tmp_path):
    script = tmp_path / "model.py"
    script.write_text(
        textwrap.dedent("""
        import multiprocessing, sys, time
        import cash

        @cash.cache(assume_safe=True)
        def work(x):
            print("RUN", x, file=sys.stderr)
            return x * x

        if __name__ == "__main__":
            with multiprocessing.get_context("spawn").Pool(2) as pool:
                print(sorted(pool.map(work, range(4))))
            print("PARENT", file=sys.stderr)
            print(work(3))
    """),
        encoding="utf-8",
    )
    run = run_python(script, cwd=tmp_path, timeout=240)
    assert run.stdout.split() == ["[0,", "1,", "4,", "9]", "9"], run.stdout
    after_pool = run.stderr.split("PARENT", 1)[1]
    assert "RUN" not in after_pool, "the parent recomputed what a worker had stored"

    listing = run_python("-m", "cash", "inspect", tmp_path / ".cash", cwd=tmp_path, check=False).stdout
    assert "model.work" in listing and "__mp_main__" not in listing, listing
