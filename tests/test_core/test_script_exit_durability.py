"""Script-context durability — async writes survive process exit.

When a user runs Cash from a Python script (not a notebook), the
process exits as soon as the script returns. ``set()`` now returns
before the actual disk write completes, so without proper ``atexit``
plumbing the data would never land. These tests spawn a real
subprocess to verify that writes survive process termination.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path


def test_decorator_writes_survive_script_exit(tmp_path: Path):
    """A script that computes a cached value, then exits cleanly, must
    leave the cache files on disk. Cash's atexit handler must drain
    the per-backend async writes before the process terminates.

    The compute is deliberately slow enough (>= 0.1s) to clear the
    default smart-persistence floor — that policy decides whether a
    value gets promoted to disk at all, and is orthogonal to the
    async-write durability we're testing here.
    """
    cache_dir = tmp_path / "cash"
    script = textwrap.dedent(f"""
        import time
        from cash import Cash
        c = Cash(cache_dir={str(cache_dir)!r}, register_magic=False)
        @c.cache
        def expensive(x):
            time.sleep(0.15)  # clear the smart-persistence 0.1s compute floor
            return x * 2
        result = expensive(123)
        assert result == 246
        # No explicit shutdown — rely on atexit.
    """)

    proc = subprocess.run(
        [sys.executable, "-c", script],
        check=True, timeout=30, capture_output=True, text=True,
    )

    # After the script exits, the cache directory must contain the files.
    assert cache_dir.exists(), f"cache dir missing: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    files = list(cache_dir.iterdir())
    data_files = [f for f in files if f.suffix == ".data"]
    meta_files = [f for f in files if f.suffix == ".meta"]
    assert data_files, f"no .data files survived script exit: {files}"
    assert meta_files, f"no .meta files survived script exit: {files}"


def test_cache_value_readable_in_second_script(tmp_path: Path):
    """The value written by script 1 must be restorable in script 2.

    This is the actual user-visible durability contract: 'I cached it in
    a previous run, the next run hits the cache instead of recomputing'.
    """
    cache_dir = tmp_path / "cash"
    sentinel = tmp_path / "computed.flag"

    # Script 1: cache a value that touches a side-effect sentinel only on
    # first computation. If the cache works across runs, the sentinel
    # stays at 1 — not 2.
    script_template = textwrap.dedent("""
        import time
        from cash import Cash
        from pathlib import Path
        c = Cash(cache_dir={cache_dir!r}, register_magic=False)
        sentinel = Path({sentinel!r})

        @c.cache
        def expensive(x):
            time.sleep(0.15)  # clear smart-persistence 0.1s floor
            n = int(sentinel.read_text()) if sentinel.exists() else 0
            sentinel.write_text(str(n + 1))
            return x * 2

        assert expensive(7) == 14
    """)

    subprocess.run(
        [sys.executable, "-c", script_template.format(
            cache_dir=str(cache_dir), sentinel=str(sentinel))],
        check=True, timeout=30,
    )
    # First run: must have computed.
    assert sentinel.read_text() == "1"

    # Second run: should hit cache, NOT recompute.
    subprocess.run(
        [sys.executable, "-c", script_template.format(
            cache_dir=str(cache_dir), sentinel=str(sentinel))],
        check=True, timeout=30,
    )
    assert sentinel.read_text() == "1", (
        "second script run recomputed — cached value did not survive exit of script 1"
    )
