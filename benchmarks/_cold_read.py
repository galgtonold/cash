"""Time a restore the way a later session performs one: in a fresh process.

Every in-process attempt to time a read of a just-written entry measured the
rig rather than the read. On one machine, the same 1 KB entry came back as:

    4.5 ms   read through the backend object that wrote it
    0.7 ms   read through a second backend created after the write
    4.9 ms   read through a second backend created before the write
    3.5-5 ms any of the above once a dozen backends are alive, because each
             one runs about eight threads
    1.5 ms   read from a new process

Only the last number is a property of cash rather than of the harness, and it is
also the one the cost model is about: the persistent tier exists to serve later
sessions, since within a session the RAM tier answers first. So the matrix times
its reads here, one process per sample.

The cost is a process start per sample (~0.8 s), which is why the process does
nothing else: it opens a backend, reads one key, and prints the seconds.
"""

from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

_READER = """
import json, sys, time
sys.path.insert(0, %(src)r)
# Pre-imported on purpose: unpickling a frame imports pandas, and in a fresh
# process that is ~450 ms of import that would land on the restore of every
# pandas and numpy family. A session restoring a frame has pandas loaded
# already -- it is how the frame got made.
for _module in ("numpy", "pandas", "scipy.sparse"):
    try:
        __import__(_module)
    except ImportError:
        pass
from cash.backends.file_backend import FileBackend
backend = FileBackend(cache_dir=%(cache)r)
start = time.perf_counter()
value = backend.get(%(key)r)
elapsed = time.perf_counter() - start
ok = value is not None and (not isinstance(value, tuple) or value[0] is not None)
print("@@" + json.dumps({"seconds": elapsed, "hit": bool(ok)}))
"""

_SRC = str(Path(__file__).resolve().parent.parent / "src")


def cold_read_seconds(cache_root: Path | str, key: str, repeats: int = 3) -> float:
    """Median seconds to restore *key* from *cache_root*, one fresh process each.

    Raises ``RuntimeError`` if a read misses: a miss is fast and would otherwise
    be recorded as an excellent restore time.
    """
    samples: list[float] = []
    with tempfile.TemporaryDirectory() as work:
        script = Path(work) / "read_one.py"
        script.write_text(_READER % {"src": _SRC, "cache": str(cache_root), "key": key}, encoding="utf-8")
        for _ in range(repeats):
            done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=900)
            if "@@" not in done.stdout:
                raise RuntimeError(f"cold read failed for {key!r}: {done.stdout[-400:]} {done.stderr[-400:]}")
            payload = json.loads(done.stdout.split("@@")[1].strip())
            if not payload["hit"]:
                raise RuntimeError(f"cold read missed for {key!r} in {cache_root}")
            samples.append(payload["seconds"])
    return statistics.median(samples)


def available() -> bool:
    """Whether a subprocess read can work here (a src/ tree beside benchmarks/)."""
    return os.path.isdir(_SRC)
