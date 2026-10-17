"""A cell's error shows the traceback IPython shows without cash.

Under cash the cell's own frame read ``File Cell In[3]:2`` where IPython
writes ``Cell In[3], line 2``, and a frame inside a function defined in an
earlier cell read ``File <cash-dfd7f3e677e4>:2, in f(a)`` at the line of its
statement rather than of its cell (``Cell In[2], line 5, in f(a)``). Each
statement is now compiled at its line in the cell, under a name IPython's
compiler labels with the cell's number.
"""

from __future__ import annotations

import re
import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("IPython")

SCRIPT = textwrap.dedent(
    """
    import contextlib, io, os, re, sys, tempfile
    os.chdir(tempfile.mkdtemp())
    from IPython.testing.globalipapp import start_ipython
    ip = start_ipython()
    if sys.argv[1] == "cash":
        with contextlib.redirect_stdout(io.StringIO()):
            ip.run_line_magic("load_ext", "cash")
            ip.run_line_magic("cash_on", "")
    cells = [
        "x = 1",
        "import math\\n\\ndef f(a):\\n    b = a + 1\\n    return b / 0",
        "y = x + 1\\nz = 2\\nf(y)",
        "1/0",
        "for i in range(2):\\n    f(i)",
    ]
    for cell in cells:
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            ip.run_cell(cell, store_history=True)
        print(re.sub(r"\\x1b\\[[0-9;]*m", "", out.getvalue()))
    """
)


def _shown(mode: str) -> str:
    done = subprocess.run([sys.executable, "-c", SCRIPT, mode], capture_output=True, text=True, timeout=240, check=True)
    lines = [line for line in done.stdout.splitlines() if line.strip() and not re.search(r"Warning|HTML object", line)]
    return "\n".join(lines)


def test_the_tracebacks_match_a_plain_run():
    plain = _shown("plain")
    assert "Cell In[2], line 5, in f(a)" in plain, plain  # control: what IPython writes
    assert _shown("cash") == plain
