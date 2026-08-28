"""What a cache can and cannot reuse when you change a grid.

Written for a question a user asked while tuning the resolution of a wave
field: *"I only made Z more refined -- why does it recompute the points I
already had?"* The intuition is reasonable and the answer is not obvious, so
this measures it rather than arguing about it.

Two workflows get conflated, and they have opposite answers:

**Revisiting** a resolution you have already run -- sweep 200, try 100, go
back to 200 -- is an instant hit and needs no technique at all. This is what
tuning a parameter actually looks like, and it is already free.

**Refining** a grid is a different thing. The axis is one argument; change how
many points it has and every value in it changes, so there is no old work to
find. Reuse is possible only when the refined axis *bitwise contains* the
coarse one, which is a property of how the axis is built, not something a
cache can add.

Each arm below reports the wall time of a second run and whether the body
executed, so "reused" is observed rather than inferred from a clock.

Usage:
    python benchmarks/bench_grid_refinement.py [--modes 200000] [--points 800]
"""
from __future__ import annotations

import argparse
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

import cash

CALLS: list[int] = []


def build_axis_linspace(n: int) -> np.ndarray:
    """The natural spelling, and the one that cannot be refined incrementally."""
    return np.linspace(0.0, 1.0, n)


def build_axis_arange(length: float, dx: float) -> np.ndarray:
    """Fixed step. Extending the domain keeps every existing point."""
    return np.arange(0.0, length, dx)


def make_field(modes: int):
    """An expensive, vectorised computation over an axis -- the shape of the
    real workload (summing many modes at each depth)."""
    mode_numbers = np.arange(1, modes + 1)[:, None]

    def field(axis: np.ndarray) -> np.ndarray:
        CALLS.append(len(axis))
        return np.sin(mode_numbers * axis[None, :]).sum(axis=0)

    return field


def timed(fn, *args):
    CALLS.clear()
    t0 = time.perf_counter()
    fn(*args)
    return time.perf_counter() - t0, list(CALLS)


def chunked(cached_chunk, axis, chunk_points):
    """Evaluate the axis in fixed-size blocks and stitch the pieces back.

    Blocks are cut by INDEX, so this only finds old work when the new axis
    starts with the old one -- see the arms below.
    """
    pieces = [cached_chunk(axis[i:i + chunk_points])
              for i in range(0, len(axis), chunk_points)]
    return np.concatenate(pieces)


def set_split(cached, axis, previous):
    """Evaluate as "the axis I had before" + "the points that are new".

    The general form: it finds old work whenever the new axis CONTAINS the old
    one, however the new points are interleaved -- which index chunking cannot
    do. The price is that the caller has to still have ``previous``, bit for
    bit, so the first call reproduces exactly.
    """
    new = np.setdiff1d(axis, previous)
    old_values = cached(previous)
    new_values = cached(new) if len(new) else np.empty(0)
    out = np.empty(len(axis), dtype=old_values.dtype)
    out[np.searchsorted(axis, previous)] = old_values
    if len(new):
        out[np.searchsorted(axis, new)] = new_values
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", type=int, default=200000)
    ap.add_argument("--points", type=int, default=800)
    ap.add_argument("--chunk", type=int, default=100)
    args = ap.parse_args()

    field = make_field(args.modes)
    root = Path(tempfile.mkdtemp(prefix="grid_bench_"))
    try:
        c = cash.Cash(cache_dir=str(root / "cache"))

        whole = c.cache(assume_safe=True)(field)
        chunk = c.cache(assume_safe=True)(field)

        n = args.points
        dx = 1.0 / n

        cold, _ = timed(whole, build_axis_linspace(n))
        print(f"cold run ({n} points, {args.modes} modes): {cold:.2f}s\n")

        def report(label, seconds, calls, note=""):
            computed = sum(calls)
            share = "nothing recomputed" if not computed else f"{computed} points recomputed"
            print(f"  {label:<44} {seconds:6.2f}s   {share}{note}")

        print("REVISITING a resolution you have run before")
        timed(whole, build_axis_linspace(n // 2))          # detour
        t, calls = timed(whole, build_axis_linspace(n))    # back again
        report("linspace(n) -> linspace(n/2) -> linspace(n)", t, calls,
               "  <- already free, no technique needed")

        print("\nREFINING, whole-axis caching")
        t, calls = timed(whole, build_axis_linspace(n + 40))
        report("linspace(n) -> linspace(n+40)", t, calls)
        t, calls = timed(whole, build_axis_linspace(2 * n - 1))
        report("linspace(n) -> linspace(2n-1)  (contains old)", t, calls)

        print("\nREFINING, index-chunked caching")
        base = build_axis_arange(1.0, dx)
        timed(chunked, chunk, base, args.chunk)            # prime
        t, calls = timed(chunked, chunk, build_axis_arange(2.0, dx), args.chunk)
        report("arange(0,1,dx) -> arange(0,2,dx)  (prefix)", t, calls,
               "  <- only the new half")
        t, calls = timed(chunked, chunk, build_axis_linspace(n + 40), args.chunk)
        report("linspace(n) -> linspace(n+40)", t, calls,
               "  <- chunking cannot save this")
        timed(chunked, chunk, build_axis_linspace(n), args.chunk)   # prime
        t, calls = timed(chunked, chunk, build_axis_linspace(2 * n - 1), args.chunk)
        report("linspace(n) -> linspace(2n-1)  (contains old)", t, calls,
               "  <- old points interleave, so blocks miss them")

        print("\nREFINING, split into (the axis I had) + (what is new)")
        split = c.cache(assume_safe=True)(field)
        coarse = build_axis_linspace(n)
        timed(split, coarse)                                        # prime
        t, calls = timed(set_split, split, build_axis_linspace(2 * n - 1), coarse)
        report("linspace(n) -> linspace(2n-1)  (contains old)", t, calls,
               "  <- only the new points")

        print("\nWhy: does the refined axis bitwise contain the coarse one?")
        for label, a, b in [
            ("linspace(n) in linspace(n+40)", build_axis_linspace(n),
             build_axis_linspace(n + 40)),
            ("linspace(n) in linspace(2n-1)", build_axis_linspace(n),
             build_axis_linspace(2 * n - 1)),
            ("arange(0,1,dx) in arange(0,2,dx)", build_axis_arange(1.0, dx),
             build_axis_arange(2.0, dx)),
        ]:
            shared = len(np.intersect1d(a, b))
            print(f"  {label:<38} {shared:5d}/{len(a)} points survive")
        return 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
