"""Decorator-shaped workload: does anything beyond GDSF pay for ``@cash.cache``?

The notebook workload re-keys statements through lineage. Decorated functions
are different: a key is ``qualname:state:dynamic:args``, calls are independent,
and what goes stale is a whole function's entries at once, when its code (its
state) changes. Each simulated day runs a few jobs, and each job calls a few
functions:

* ``pipeline`` jobs call a function over its working set of arguments (the
  same ones every day, drifting slowly), like a nightly job or a notebook's
  ``@cash.cache`` helpers;
* ``sweep`` jobs call it over arguments drawn Zipf-style from a large space,
  like a parameter search, so most calls are one-offs.

A code edit bumps a function's version: every one of its keys changes.

The policy under test is offered the entry's slot ``(function, args)``, which is
what a supersede rule for decorators would have to use -- and that is only
IDEAL here. In cash that slot is unsafe (closures from one factory, bound
``self``, globals and seeds all share it), so if even the ideal rule does not
beat GDSF, the unsafe one is not worth building.

    python benchmarks/eviction_sim/decorator_workload.py --seeds 5
"""

from __future__ import annotations

import argparse
import math
import random
import statistics
from dataclasses import dataclass, replace

import policies as P


@dataclass
class DParams:
    n_functions: int = 12
    days: int = 60
    jobs_per_day: tuple = (2, 6)
    p_sweep_job: float = 0.3
    working_set: tuple = (5, 200)  # args a pipeline job calls
    sweep_calls: tuple = (50, 400)
    arg_space: int = 50_000  # sweep draws Zipf over this
    zipf_s: float = 1.1
    p_edit_per_day: float = 0.08  # per function
    p_drift: float = 0.05  # per working-set arg per day
    compute_median: float = 0.4
    compute_sigma: float = 1.6
    size_small_median: float = 5e3
    size_large_median: float = 40e6
    p_large: float = 0.25
    restore_bps: float = 400e6
    restore_floor: float = 0.002
    seed: int = 0


class DProject:
    def __init__(self, p: DParams):
        self.p = p
        rng = self.rng = random.Random(p.seed)
        self.fn = []
        for f in range(p.n_functions):
            compute = p.compute_median * math.exp(p.compute_sigma * rng.gauss(0, 1))
            large = rng.random() < p.p_large
            size = (p.size_large_median if large else p.size_small_median) * math.exp(1.2 * rng.gauss(0, 1))
            ws = rng.randint(*p.working_set)
            self.fn.append(
                {
                    "compute": compute,
                    "size": int(max(64, size)),
                    "version": 0,
                    "working": list(range(ws)),
                    "next_arg": ws,
                }
            )
        # Zipf CDF over the sweep arg space, shared
        weights = [1 / (i + 1) ** p.zipf_s for i in range(p.arg_space)]
        total = sum(weights)
        acc, self.cdf = 0.0, []
        for w in weights:
            acc += w / total
            self.cdf.append(acc)

    def zipf(self, rng):
        import bisect

        return bisect.bisect_left(self.cdf, rng.random())

    def entry(self, f, arg, rng_jitter=None):
        fn = self.fn[f]
        # per-arg variation in cost and size, deterministic in (f, arg)
        r = random.Random(hash((f, arg)))
        compute = fn["compute"] * math.exp(0.5 * r.gauss(0, 1))
        size = int(fn["size"] * math.exp(0.5 * r.gauss(0, 1)))
        key = hash((f, fn["version"], arg))
        return key, max(64, size), compute, (f, arg)


def session(proj: DProject):
    p = proj.p
    rng = random.Random(p.seed + 1)
    for _day in range(p.days):
        for f, fn in enumerate(proj.fn):
            if rng.random() < p.p_edit_per_day:
                fn["version"] += 1
            fn["working"] = [
                a if rng.random() > p.p_drift else (fn.__setitem__("next_arg", fn["next_arg"] + 1) or fn["next_arg"])
                for a in fn["working"]
            ]
        for _job in range(rng.randint(*p.jobs_per_day)):
            f = rng.randrange(p.n_functions)
            if rng.random() < p.p_sweep_job:
                args = [10_000_000 + proj.zipf(rng) for _ in range(rng.randint(*p.sweep_calls))]
            else:
                args = list(proj.fn[f]["working"])
            for a in args:
                yield proj.entry(f, a)


def run(p: DParams, pol):
    proj = DProject(p)
    t = 0.0
    for key, size, compute, slot in session(proj):
        restore = p.restore_floor + size / p.restore_bps
        if pol.access(key, size, compute - restore, {"slot": slot}):
            t += restore
        else:
            t += compute
            if compute >= 0.1 and compute - restore > 0.2 * compute:
                pol.insert(key, size, compute - restore, {"slot": slot})
    return t, pol


def live_bytes(p: DParams):
    """Bytes of the current-version entries of each function's working set
    that an unlimited cache holds at the end -- what a rerun would want."""
    t, unl = run(p, P.LRU(float("inf")))
    proj = DProject(p)
    for _ in session(proj):
        pass
    live = 0
    for f, fn in enumerate(proj.fn):
        for a in fn["working"]:
            key, size, _c, _s = proj.entry(f, a)
            if key in unl.size:
                live += size
    return t, unl.used, live


POL = {
    "LRU": lambda c: P.LRU(c),
    "GDSF": lambda c: P.GDSF(c),
    "Supersede+GDSF-noage (ideal slots)": lambda c: P.SupersedeAware(c, keep=1, inner=P.GDSF, age_on_dead=False),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--fracs", default="0.5,1,2,4")
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    base = DParams()
    import ast

    for kv in a.set:
        k, v = kv.split("=", 1)
        base = replace(base, **{k: type(getattr(base, k))(ast.literal_eval(v))})
    fracs = [float(x) for x in a.fracs.split(",")]
    res = {n: {f: [] for f in fracs} for n in POL}
    fps = []
    for seed in range(a.seeds):
        prm = replace(base, seed=seed)
        t_inf, footprint, live = live_bytes(prm)
        t_none, _ = run(prm, P.LRU(0))
        fps.append(footprint / max(1, live))
        for f in fracs:
            for n, mk in POL.items():
                t, _pol = run(prm, mk(int(live * f)))
                res[n][f].append((t - t_inf) / max(1e-9, t_none - t_inf))
    print(
        f"[decorator workload] lost savings, mean over {a.seeds} seeds; cap = multiple of live bytes "
        f"(uncapped footprint {statistics.mean(fps):.1f}x live)"
    )
    print(f"{'policy':<38}" + "".join(f"{str(f) + 'x':>9}" for f in fracs))
    for n in POL:
        print(f"{n:<38}" + "".join(f"{statistics.mean(res[n][f]) * 100:8.1f}%" for f in fracs))


if __name__ == "__main__":
    main()
