"""What does a cache cost per operation as it fills, and does the backend matter?

Three questions, which is what the table has to answer before anyone decides
whether to shard the file layout, change the default, or leave both alone:

1. **The first write into an empty cache** -- the floor.
2. **A write once N entries are already there** -- does it degrade, and how fast.
3. **The same for reads and for opening the cache in a fresh process.**

Arms are INTERLEAVED per round: file, sqlite, file, sqlite. Sequential arms
would charge whichever backend happened to run during background load, and
this repo has been bitten by exactly that before. Medians, not means, for the
same reason.

Writes are drained before and after timing. ``set`` returns once the write is
queued, so timing it alone measures the queue (~6us) rather than the disk; and
draining only afterwards measures the whole accumulated backlog, which is how
an earlier version of this reported 3.5 SECONDS per write.

Usage:
    python benchmarks/bench_backend_scale.py [--counts 0,1000,5000,20000]
"""
from __future__ import annotations

import argparse
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROUNDS = 7


def make_backend(kind: str, root: Path):
    from cash.backends import FileBackend
    from cash.backends.sqlite_backend import SQLiteBackend
    if kind == "file":
        return FileBackend(str(root))
    return SQLiteBackend(str(root / "cache.db"))


def _meta():
    now = time.time()
    return {"size": 512, "created_at": now, "last_access": now}


def drain(backend) -> None:
    writes = getattr(backend, "_writes", None)
    if writes is not None:
        writes.wait_all()


def timed_write(backend, key: str) -> float:
    drain(backend)                       # clear the backlog BEFORE timing
    t = time.perf_counter()
    backend.set(key, b"x" * 512, _meta())
    drain(backend)                       # ...and drain only this one
    return time.perf_counter() - t


def timed_read(backend, key: str) -> float:
    t = time.perf_counter()
    backend.get(key)
    return time.perf_counter() - t


def open_cost(kind: str, root: Path) -> float:
    """First operation in a FRESH process -- what a script pays at startup.

    Measured twice, second reported: the first read of a newly written file
    costs ~13ms on Windows (Defender) against ~0.13ms warm, and that tax
    belongs to the machine, not the backend.
    """
    script = (
        "import sys, time\n"
        "sys.path.insert(0, sys.argv[3])\n"
        "from cash.backends import FileBackend\n"
        "from cash.backends.sqlite_backend import SQLiteBackend\n"
        "kind, root = sys.argv[1], sys.argv[2]\n"
        "b = FileBackend(root) if kind == 'file' else SQLiteBackend(root + '/cache.db')\n"
        "t = time.perf_counter(); b.get('k0'); print(time.perf_counter() - t)\n"
    )
    last = 0.0
    for _ in range(2):
        out = subprocess.run([sys.executable, "-c", script, kind, str(root), str(REPO / "src")],
                             capture_output=True, text=True, cwd=str(REPO))
        if out.returncode != 0:
            print(out.stderr[-1200:])
            raise SystemExit(1)
        last = float(out.stdout.strip().splitlines()[-1])
    return last


def dir_stats(root: Path) -> tuple[int, int]:
    """Files and bytes on disk.

    Counts SQLite's ``-wal`` and ``-shm`` alongside the ``.db``: until a
    checkpoint runs those hold real, unrecoverable-without-them data, and
    reporting only the ``.db`` made an unflushed cache look like 0 bytes.
    """
    files = [f for f in root.rglob("*") if f.is_file()]
    return len(files), sum(f.stat().st_size for f in files)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default="0,1000,5000,20000")
    args = ap.parse_args()
    counts = [int(c) for c in args.counts.split(",")]

    kinds = ("file", "sqlite")
    root = Path(tempfile.mkdtemp(prefix="cash_backend_scale_"))
    try:
        roots = {k: root / k for k in kinds}
        for r in roots.values():
            r.mkdir(parents=True)
        backends = {k: make_backend(k, roots[k]) for k in kinds}
        filled = {k: 0 for k in kinds}

        print(f"  {'entries':>8}  {'backend':<8}{'write':>10}{'read':>10}"
              f"{'open':>11}{'files':>10}{'on disk':>11}{'per entry':>12}")
        for n in counts:
            # Fill both arms to N before measuring either.
            for k in kinds:
                while filled[k] < n:
                    backends[k].set(f"k{filled[k]}", b"x" * 512, _meta())
                    filled[k] += 1
                drain(backends[k])

            writes = {k: [] for k in kinds}
            reads = {k: [] for k in kinds}
            for r in range(ROUNDS):
                for k in kinds:                       # interleaved
                    writes[k].append(timed_write(backends[k], f"probe_{n}_{r}"))
                    if n:
                        reads[k].append(timed_read(backends[k], f"k{n // 2}"))

            for k in kinds:
                files, size = dir_stats(roots[k])
                w = statistics.median(writes[k]) * 1000
                rd = statistics.median(reads[k]) * 1000 if reads[k] else float("nan")
                op = open_cost(k, roots[k]) * 1000
                rd_s = "     -" if rd != rd else f"{rd:>7.3f}ms"
                print(f"  {n:>8,}  {k:<8}{w:>8.2f}ms{rd_s:>10}{op:>9.1f}ms"
                      f"{files:>10,}{size/1e6:>9.1f}MB"
                      f"{(size / n if n else 0):>10,.0f}B/e")
            print()
        return 0
    finally:
        for b in list(backends.values()):
            close = getattr(b, "close", None)
            if close:
                try:
                    close()
                except Exception:  # noqa: BLE001 - teardown
                    pass
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
