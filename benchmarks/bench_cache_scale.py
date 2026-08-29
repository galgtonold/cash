"""How does a cache directory behave at 10k, 100k, a million entries?

Collisions are one scale worry and a small one. The other is mechanical: every
entry is two files in ONE flat directory, and several operations walk all of
them. This measures which costs grow with the entry count and which do not, so
the fix (if any) is aimed at something measured.

What to watch:

* **First access** pays ``_init_stats()``, which reads every ``.meta`` in the
  directory to total the size and seed the LRU. That is per PROCESS, so a
  script that runs for two seconds pays it in full every time.
* **Steady-state get/set** should be flat: the filename is derived from the
  key, so the lookup is one ``open()`` and the filesystem's own directory
  index does the work.
* **``cash inspect``** is deliberately O(N) -- it exists to summarise the whole
  directory -- but it should stay usable at the sizes people reach.

Usage:
    python benchmarks/bench_cache_scale.py [--counts 1000,10000,50000]
"""
from __future__ import annotations

import argparse
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def build_directory(cache_dir: Path, n: int, payload: bytes) -> float:
    """Write *n* entries the way FileBackend lays them out, without the backend.

    Writing them through ``set()`` would fold the backend's own per-write costs
    into the fixture and make the measurement circular; this isolates the
    directory shape from the code under test.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "CACHE_VERSION").write_text("1", encoding="utf-8")
    t0 = time.perf_counter()
    for i in range(n):
        stem = f"{i:064x}"
        meta = {"key": f"mod.f:state:{i}:args", "func_name": "mod.f",
                "size": len(payload), "created_at": time.time(),
                "last_access": time.time(), "access_count": 1,
                "execution_time": 0.5, "storage": ["DISK"]}
        (cache_dir / f"{stem}.meta").write_bytes(pickle.dumps(meta))
        (cache_dir / f"{stem}.data").write_bytes(payload)
    return time.perf_counter() - t0


def time_first_access(cache_dir: Path) -> float:
    """Cost of the first operation on a fresh backend -- i.e. _init_stats()."""
    script = (
        "import sys, time\n"
        "from cash.backends import FileBackend\n"
        "b = FileBackend(sys.argv[1])\n"
        "t = time.perf_counter()\n"
        "b.get('mod.f:state:0:args')\n"          # forces _ensure_initialized
        "print(time.perf_counter() - t)\n"
    )
    out = subprocess.run([sys.executable, "-c", script, str(cache_dir)],
                         capture_output=True, text=True, cwd=str(REPO))
    if out.returncode != 0:
        print(out.stderr[-1500:])
        raise SystemExit(1)
    return float(out.stdout.strip().splitlines()[-1])


def time_steady_state(cache_dir: Path, n: int) -> tuple[float, float]:
    """get/set latency once the backend is warm."""
    script = (
        "import sys, time, statistics\n"
        "from cash.backends import FileBackend\n"
        "b = FileBackend(sys.argv[1]); n = int(sys.argv[2])\n"
        "b.get('warm')\n"
        "gets = []\n"
        "for i in range(0, n, max(1, n // 50)):\n"
        "    t = time.perf_counter(); b.get(f'mod.f:state:{i}:args')\n"
        "    gets.append(time.perf_counter() - t)\n"
        "sets = []\n"
        "for i in range(50):\n"
        "    t = time.perf_counter(); b.set(f'new:{i}', b'x' * 512)\n"
        "    sets.append(time.perf_counter() - t)\n"
        "print(statistics.median(gets), statistics.median(sets))\n"
    )
    out = subprocess.run([sys.executable, "-c", script, str(cache_dir), str(n)],
                         capture_output=True, text=True, cwd=str(REPO))
    if out.returncode != 0:
        print(out.stderr[-1500:])
        raise SystemExit(1)
    g, s = out.stdout.strip().splitlines()[-1].split()
    return float(g), float(s)


def time_inspect(cache_dir: Path) -> float:
    """Second run, for the same reason `first access` is measured twice.

    `cash inspect` reads every .meta by design. On this machine the FIRST read
    of a file costs ~13 ms (Defender) against ~0.13 ms warm, so a cold run
    measures the antivirus rather than the command -- 269 s versus 2 s at 20k
    entries. The warm number is what a user with an existing cache sees.
    """
    def once() -> float:
        t = time.perf_counter()
        subprocess.run([sys.executable, "-m", "cash", "inspect", str(cache_dir)],
                       capture_output=True, text=True, cwd=str(REPO))
        return time.perf_counter() - t
    once()
    return once()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", default="1000,10000,50000")
    ap.add_argument("--payload", type=int, default=512,
                    help="bytes per entry (default 512)")
    args = ap.parse_args()
    counts = [int(c) for c in args.counts.split(",")]
    payload = b"x" * args.payload

    root = Path(tempfile.mkdtemp(prefix="cash_scale_"))
    try:
        print(f"payload {args.payload} B per entry, flat directory\n")
        print(f"  {'entries':>9}{'build':>9}{'1st access':>13}{'get':>11}"
              f"{'set':>11}{'cash inspect':>15}{'dir bytes':>13}")
        for n in counts:
            cache_dir = root / f"c{n}"
            build = build_directory(cache_dir, n, payload)
            # The FIRST read of a newly written file costs ~13 ms on this
            # machine (Defender scanning it) against ~0.13 ms warm -- 100x, and
            # it would land on cash's ledger. `_init_stats` reads every .meta,
            # so running it twice warms them for free; the SECOND number is the
            # per-process cost a real user pays, where the scan already
            # happened at write time.
            time_first_access(cache_dir)          # discard: pays the AV tax
            first = time_first_access(cache_dir)
            get, put = time_steady_state(cache_dir, n)
            insp = time_inspect(cache_dir)
            total = sum(f.stat().st_size for f in cache_dir.iterdir())
            print(f"  {n:>9,}{build:>8.1f}s{first*1000:>11.1f} ms"
                  f"{get*1e6:>8.0f} us{put*1e6:>8.0f} us"
                  f"{insp:>13.2f} s{total/1e6:>10.1f} MB")
            shutil.rmtree(cache_dir, ignore_errors=True)
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
