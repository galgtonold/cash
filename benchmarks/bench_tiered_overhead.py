"""What does the DEFAULT backend cost, as opposed to a bare one?

Every other storage benchmark in this directory measures a bare backend, and
nobody runs a bare backend: the factory builds ``TieredBackend([RAM, disk])``.
So the per-call overhead of the tier wrapper itself -- the promotion policy on
every write, the read-repair on every L2 hit -- has never been on a ledger.

Four arms, interleaved per round:

* ``memory``  -- InMemoryBackend alone, the floor.
* ``file``    -- FileBackend alone, what the other benchmarks report.
* ``tiered``  -- the real default stack.
* ``tiered-nopromote`` -- the same stack with an entry too cheap to persist,
  which isolates the cost of *deciding* not to promote from the cost of
  promoting.

and three operations, because the tiers make them behave differently:

* ``set``      -- write.
* ``get L1``   -- a hit in RAM. The common case once a notebook is warm.
* ``get L2``   -- a hit on disk with RAM cold, which is a kernel restart. This
  one also pays read-repair: ``TieredBackend.get`` promotes the value back
  into every faster tier before returning it, synchronously.

Usage:
    python benchmarks/bench_tiered_overhead.py [--payload 512] [--rounds 15]
"""
from __future__ import annotations

import argparse
import shutil
import statistics
import tempfile
import time
from pathlib import Path

from cash.backends import FileBackend, InMemoryBackend, TieredBackend

# The factory's smart-persistence stack lowers the compute floor to this.
COMPUTE_FLOOR_S = 0.1


def _meta(payload_len: int, *, promote: bool) -> dict:
    now = time.time()
    return {
        "size": payload_len,
        "created_at": now,
        "last_access": now,
        # Above the floor -> the disk tier accepts it; below -> RAM only.
        "execution_time": 1.0 if promote else 0.0,
    }


def drain(backend) -> None:
    """Settle every async write in the stack, including each child tier's."""
    for b in getattr(backend, "backends", [backend]):
        writes = getattr(b, "_writes", None)
        if writes is not None:
            writes.wait_all()


def build(kind: str, root: Path):
    if kind == "memory":
        return InMemoryBackend(max_entries=100_000)
    if kind == "file":
        return FileBackend(str(root))
    return TieredBackend(
        [InMemoryBackend(max_entries=100_000), FileBackend(str(root))],
        min_persist_compute_s=COMPUTE_FLOOR_S,
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--payload", type=int, default=512)
    ap.add_argument("--rounds", type=int, default=15)
    args = ap.parse_args()

    payload = b"x" * args.payload
    arms = ("memory", "file", "tiered", "tiered-nopromote")
    root = Path(tempfile.mkdtemp(prefix="cash_tiered_"))
    backends = {}
    try:
        for a in arms:
            (root / a).mkdir(parents=True)
            backends[a] = build("tiered" if a.startswith("tiered") else a, root / a)

        # One seeded key per arm for the read measurements.
        for a, b in backends.items():
            b.set("seed", payload, _meta(len(payload), promote=True))
            drain(b)

        results = {a: {"set": [], "get L1": [], "get L2": []} for a in arms}
        for r in range(args.rounds):
            for a in arms:                                   # interleaved
                b = backends[a]
                promote = not a.endswith("nopromote")

                drain(b)
                t = time.perf_counter()
                b.set(f"k{r}", payload, _meta(len(payload), promote=promote))
                drain(b)
                results[a]["set"].append(time.perf_counter() - t)

                t = time.perf_counter()
                b.get("seed")
                results[a]["get L1"].append(time.perf_counter() - t)

                # Cold RAM: only meaningful for the stack. For the bare arms
                # this repeats the measurement above, which is the honest
                # comparison -- they have no second tier to fall through to.
                if isinstance(b, TieredBackend):
                    b.backends[0].delete("seed")
                t = time.perf_counter()
                b.get("seed")
                drain(b)                                     # read-repair write
                results[a]["get L2"].append(time.perf_counter() - t)

        print()
        print(f"  payload = {args.payload:,}B, {args.rounds} rounds, median")
        print()
        print(f"  {'arm':<20}{'set':>12}{'get L1':>12}{'get L2':>12}")
        base = {}
        for a in arms:
            row = [statistics.median(results[a][op]) * 1000
                   for op in ("set", "get L1", "get L2")]
            if a == "file":
                base = dict(zip(("set", "get L1", "get L2"), row))
            print(f"  {a:<20}" + "".join(f"{v:>10.3f}ms" for v in row))

        if base:
            print()
            print("  tiered vs bare file:")
            row = {op: statistics.median(results["tiered"][op]) * 1000
                   for op in ("set", "get L1", "get L2")}
            for op in ("set", "get L1", "get L2"):
                delta = row[op] - base[op]
                print(f"    {op:<8}{delta:+8.3f}ms  ({row[op] / base[op]:.2f}x)")
        return 0
    finally:
        for b in backends.values():
            try:
                b.shutdown()
            except Exception:  # noqa: BLE001 - teardown
                pass
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
