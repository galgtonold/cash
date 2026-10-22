"""A notebook on a cache with no local folder (SQLite here; Redis and S3
alike) checks a hit by its own entry's trace, never by another notebook's.

Traces were kept in a ``shadow`` folder beside the cache, and a cache with
no folder used one under the temp dir, shared by every such notebook on the
machine and keyed by cache key alone. A statement that finds its function by
a string (``globals()[name](3)``) keys on the name, not the code: notebook
B's trace for that key replaced A's, and once A's function was edited to
match B's, A's old entry was "agreed" and served (t=4 where the answer is
8). Each entry now carries its own trace.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.version_info < (3, 12), reason="traces are recorded on 3.12+")

KERNEL = r"""
import json, os, sys
from unittest.mock import MagicMock
from traitlets.config import Configurable
from cash import Cash
from cash.analysis.annotations import CacheAnnotation
from cash.backends.sqlite_backend import SQLiteBackend
from cash.notebook import shadow
from cash.notebook.ipython.magics import CashMagics

class Pub:
    def publish(self, *a, **k):
        pass

class Shell(Configurable):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.user_ns = {"_ih": [], "__name__": "__main__"}
        self.user_global_ns = self.user_ns
        self.events = MagicMock(); self.ast_transformers = []; self.input_transformers_cleanup = []
        self.run_cell = MagicMock(); self.display_pub = Pub()
        self.magics_manager = MagicMock(); self.magics_manager.magics = {"cell": {}, "line": {}}

folder, factor = sys.argv[1], int(sys.argv[2])
shell = Shell()
cash = Cash(backend=SQLiteBackend(os.path.join(folder, "c.db")), register_magic=False)
proc = CashMagics(shell, cash)._statement_processor
shadow.disable()
shadow.enable()
out = {}
for code in ("cfg = 1", "name = 'make'", f"def make(x):\n    return x + cfg * {factor}", "t = globals()[name](3)"):
    out[code.splitlines()[0]] = str(proc.process_statement(code, annotation=CacheAnnotation(persist=True))["status"])
out["t"] = shell.user_ns["t"]
print(json.dumps(out))
"""


def _notebook(folder, factor, env):
    os.makedirs(folder, exist_ok=True)
    done = subprocess.run(
        [sys.executable, "-c", KERNEL, str(folder), str(factor)],
        capture_output=True,
        text=True,
        timeout=180,
        env=env,
        check=True,
    )
    return json.loads(done.stdout.strip().splitlines()[-1])


def test_another_notebooks_trace_does_not_vouch_for_this_ones_entry(tmp_path):
    env = {**os.environ, "TMPDIR": str(tmp_path / "tmp"), "PYTHONPATH": os.pathsep.join(sys.path)}
    env.pop("CASH_CACHE_DIR", None)
    env.pop("CASH_SHADOW", None)
    env.pop("CASH_TRACE", None)
    os.makedirs(tmp_path / "tmp")
    a, b = tmp_path / "a", tmp_path / "b"

    assert _notebook(a, 1, env)["t"] == 4
    assert _notebook(b, 5, env)["t"] == 8
    again = _notebook(a, 5, env)

    assert again["t"] == 8, again
