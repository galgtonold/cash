"""`@cash.cache` runs with the notebook package unimportable.

The decorator shares file tracking, randomness detection, static analysis and
object hashing with the notebook path. That shared code lives outside
``cash.notebook``, and only the magics loaders import the notebook package, so
a cached function must work end to end (a miss, an in-memory hit, a disk hit in
a new process) while any import of ``cash.notebook`` fails.
"""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

from tests._scripts import run_python

_BLOCKER = """
import sys

class _BlockNotebook:
    def find_spec(self, name, path=None, target=None):
        if name == "cash.notebook" or name.startswith("cash.notebook."):
            raise ImportError(f"blocked by the test: {name}")
        return None

sys.meta_path.insert(0, _BlockNotebook())
"""

_SCRIPT = """
import os
import random
import sys
import time

try:
    import cash.notebook  # noqa: F401 - imported for its side effects, if present
except ImportError:
    pass
else:
    raise SystemExit("the blocker let cash.notebook through")

from cash import Cash, FileBackend

c = Cash(backend=FileBackend(cache_dir=sys.argv[1]), register_magic=False)
DATA = sys.argv[2]
RUNS = sys.argv[3]


@c.cache
def load(scale):
    fd = os.open(RUNS, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    os.write(fd, b"x\\n")
    os.close(fd)
    random.seed(scale)
    with open(DATA) as f:
        return [scale * int(v) + random.randint(0, 0) for v in f.read().split()]


first = load(3)
second = load(3)
assert first == second == [3, 6, 9], (first, second)
leaked = sorted(m for m in sys.modules if m == "cash.notebook" or m.startswith("cash.notebook."))
assert not leaked, leaked
print("OK")
"""


def _run(tmp_path: Path, script: Path) -> subprocess.CompletedProcess:
    return run_python(script, tmp_path / "cache", tmp_path / "data.txt", tmp_path / "runs", cwd=tmp_path, check=False)


def test_a_cached_function_runs_without_the_notebook_package(tmp_path):
    (tmp_path / "data.txt").write_text("1 2 3\n", encoding="utf-8")
    script = tmp_path / "use_cash.py"
    script.write_text(
        textwrap.dedent(_BLOCKER) + textwrap.dedent(_SCRIPT),
        encoding="utf-8",
    )

    first = _run(tmp_path, script)
    assert first.returncode == 0 and "OK" in first.stdout, first.stderr
    # A new process restores from disk: the body ran once across both.
    second = _run(tmp_path, script)
    assert second.returncode == 0 and "OK" in second.stdout, second.stderr
    assert (tmp_path / "runs").read_text(encoding="utf-8") == "x\n"
