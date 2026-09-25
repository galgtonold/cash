"""A cached generator left unfinished when the script ends exits silently.

The stream holds an observation scope open while it is suspended. A script
that ends with one still referenced has it closed while the interpreter tears
its modules down, and putting the reader patches back then failed on
``sys.meta_path`` being None: "Exception ignored in: <generator object
ResultStore.stream_and_store ...>" and a traceback, where an undecorated
generator exits without a word.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

PROGRAM = textwrap.dedent("""
    import cash
    c = cash.Cash(cache_dir=CACHE)

    @c.cache(chunk_max_items=2)
    def lines(n):
        yield from range(n)

    g = lines(10)
    print(next(g), next(g), next(g))   # a chunk is written; the stream is left open
""")


@pytest.mark.timeout(300)
def test_no_traceback_at_exit(tmp_path):
    script = tmp_path / "run.py"
    script.write_text(PROGRAM.replace("CACHE", repr(str(tmp_path / ".cash"))), encoding="utf-8")
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=180, cwd=str(tmp_path))
    assert done.returncode == 0, done.stderr[-1500:]
    assert done.stdout.split() == ["0", "1", "2"]
    assert "Traceback" not in done.stderr and "Exception ignored" not in done.stderr, done.stderr
