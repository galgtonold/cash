"""Sweep policies x cache sizes x seeds over the notebook workload.

Two experiments, each isolating one tier:

  --tier disk : RAM tier fixed (cash's current RAM policy at --ram-cap x live),
                disk policy varies, disk cap swept.
  --tier ram  : disk tier unlimited, RAM policy varies, RAM cap swept.

Metric: *lost savings* of the tier under test
    (T_policy - T_unlimited) / (T_absent - T_unlimited)
0% = as good as an unlimited tier, 100% = as bad as not having the tier.
Caps are multiples of the final live-set bytes (what a run-all of every
notebook's current source would request).

    python benchmarks/eviction_sim/run.py --tier disk --seeds 5 -q
    python benchmarks/eviction_sim/run.py --tier ram --policies cash-RAM-now,LRU,GDSF
    python benchmarks/eviction_sim/run.py --set p_undo=0.4 --set days=120
    python benchmarks/eviction_sim/run.py --list
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import statistics
import sys
import time
from dataclasses import replace

import policies as P
from workload import Engine, Params, Project, session_ops


class Unlimited(P.LRU):
    name = "unlimited"

    def __init__(self, cap=None):
        super().__init__(float("inf"))


class NoCache(P.Policy):
    name = "none"

    def __init__(self, cap=None):
        super().__init__(0)

    def insert(self, *a, **k):
        pass


def simulate(params: Params, disk, ram, live_hints=False, touch_hints=False, liveset_hints=False):
    proj = Project(params)
    eng = Engine(proj, disk, ram)
    frontier = None
    for op in session_ops(proj):
        kind = op[0]
        if kind == "restart":
            eng.restart()
        elif kind == "run":
            if touch_hints:
                # cash's upstream simulator computes the current key of every
                # statement above the cell before it runs; loop iterations it
                # cannot enumerate without executing, so they are not touched.
                keys = [proj.key(sid) for sid in proj.nb_stmts[op[1]]
                        if proj.stmts[sid].cell <= op[2] and not proj.stmts[sid].loop_iters]
                for pol in (disk, ram):
                    if pol is not None:
                        pol.hint("touch", keys=keys)
            if liveset_hints and frontier is not None:
                # the notebook's FULL live set: every top-level statement of
                # its current source (loop iterations are not enumerable)
                nb = op[1]
                keys = {proj.key(sid) for sid in proj.nb_stmts[nb]
                        if proj.stmts[sid].cell < frontier[nb] and not proj.stmts[sid].loop_iters}
                disk.hint("liveset", nb=nb, keys=keys)
            eng.run_cell(op[1], op[2])
        elif kind in ("frontier", "edited"):
            if kind == "frontier":
                frontier = op[1]
            if live_hints and frontier is not None:
                for pol in (disk, ram):
                    if pol is not None:
                        pol.hint("live", keys=proj.live_keys(frontier))
    eng.frontier = frontier
    return eng


def live_bytes(params: Params):
    """(unbounded disk footprint, disk live bytes, engine) for one seed.

    The engine carries ``ram_live``: the live bytes the RAM tier would admit.
    """
    disk = Unlimited()
    eng = simulate(params, disk, None)
    live = eng.proj.live_keys(eng.frontier)
    # only what the disk tier actually admitted counts toward its live set
    lb = sum(disk.size[k] for k in live if k in disk.size)
    # RAM-admissible live bytes (Gate 0 + Gate A), for the RAM experiment
    rb = 0
    for sid in range(len(eng.proj.stmts)):
        for key, size, compute, slot in eng.proj.entries(sid):
            if key in live and compute >= 0.01 and eng.ram_cost(size) <= max(0.05, 0.8 * compute):
                rb += size
    eng.ram_live = rb
    return disk.used, lb, eng


POLICIES = {
    "cash-now": lambda cap: P.CashCurrent(cap),
    "cash-nocrumb": lambda cap: P.CashCurrent(cap, crumbs=False),
    "cash-nowm": lambda cap: P.CashCurrent(cap, target=1.0),
    "cash-RAM-now": lambda cap: P.CashRAM(cap),
    "LRU": lambda cap: P.LRU(cap),
    "FIFO": lambda cap: P.FIFO(cap),
    "LFU": lambda cap: P.LFU(cap),
    "SLRU": lambda cap: P.SLRU(cap),
    "ARC": lambda cap: P.ARC(cap),
    "S3-FIFO": lambda cap: P.S3FIFO(cap),
    "SIEVE": lambda cap: P.SIEVE(cap),
    "SRRIP": lambda cap: P.RRIP(cap, "S"),
    "BRRIP": lambda cap: P.RRIP(cap, "B"),
    "DRRIP": lambda cap: P.RRIP(cap, "D"),
    "GD-Size": lambda cap: P.GDSF(cap, use_freq=False),
    "GDSF": lambda cap: P.GDSF(cap),
    "CostLRU": lambda cap: P.CostLRU(cap, tau=2000),
    "CostLRU-f": lambda cap: P.CostLRU(cap, tau=2000, freq=True),
    "Supersede+LRU": lambda cap: P.SupersedeAware(cap, keep=1),
    "Supersede+GDSF": lambda cap: P.SupersedeAware(cap, keep=1, inner=P.GDSF),
    "Supersede+CostLRU": lambda cap: P.SupersedeAware(cap, keep=1, inner=P.CostLRU, inner_kw={"tau": 2000}),
    "LiveGC+LRU": lambda cap: P.LiveSetGC(cap),
    "LiveGC+GDSF": lambda cap: P.LiveSetGC(cap, inner=P.GDSF),
    "LiveGC+CostLRU": lambda cap: P.LiveSetGC(cap, inner=P.CostLRU, inner_kw={"tau": 2000}),
    "Supersede0+CostLRU": lambda cap: P.SupersedeAware(cap, keep=0, inner=P.CostLRU, inner_kw={"tau": 2000}),
    "Supersede2+CostLRU": lambda cap: P.SupersedeAware(cap, keep=2, inner=P.CostLRU, inner_kw={"tau": 2000}),
    "Hybrid+CostLRU": lambda cap: P.HybridGC(cap, keep=1, inner=P.CostLRU, inner_kw={"tau": 2000}),
    "Hybrid+GDSF": lambda cap: P.HybridGC(cap, keep=1, inner=P.GDSF),
    "CostLRU-t500": lambda cap: P.CostLRU(cap, tau=500),
    "Supersede+GDSF-noage": lambda cap: P.SupersedeAware(cap, keep=1, inner=P.GDSF, age_on_dead=False),
    "Supersede0+GDSF-noage": lambda cap: P.SupersedeAware(cap, keep=0, inner=P.GDSF, age_on_dead=False),
    "Supersede2+GDSF-noage": lambda cap: P.SupersedeAware(cap, keep=2, inner=P.GDSF, age_on_dead=False),
    "Supersede+CostLRU-t500": lambda cap: P.SupersedeAware(cap, keep=1, inner=P.CostLRU, inner_kw={"tau": 500}),
    "CostLRU-t8000": lambda cap: P.CostLRU(cap, tau=8000),
    "GDSF+touch": lambda cap: P.TouchedGDSF(cap),
    "OwnerGC-g5": lambda cap: P.OwnerGC(cap, grace=5),
    "OwnerGC-g20": lambda cap: P.OwnerGC(cap, grace=20),
    "OwnerGC-g100": lambda cap: P.OwnerGC(cap, grace=100),
    "GDSF-s8": lambda cap: P.SampledGDSF(cap, k=8),
    "GDSF-s32": lambda cap: P.SampledGDSF(cap, k=32),
    "GDSF-s128": lambda cap: P.SampledGDSF(cap, k=128),
    "Supersede+GDSF-s32": lambda cap: P.SupersedeAware(cap, keep=1, inner=P.SampledGDSF, inner_kw={"k": 32},
                                                        age_on_dead=False),
}
NEEDS_LIVE = {n for n in POLICIES if n.startswith(("LiveGC", "Hybrid"))}
NEEDS_TOUCH = {"GDSF+touch"}
NEEDS_LIVESET = {n for n in POLICIES if n.startswith("OwnerGC")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["disk", "ram"], default="disk")
    ap.add_argument("--ram-cap", type=float, default=0.25,
                    help="disk experiment: RAM cap as multiple of live bytes")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--fracs", default="0.25,0.5,1,2,4")
    ap.add_argument("--policies", default=None, help="comma-separated names (see --list)")
    ap.add_argument("--set", action="append", default=[], help="Params override, e.g. p_undo=0.4")
    ap.add_argument("--out", default=None,
                    help="write raw per-seed results as JSON (benchmarks/results_* is gitignored)")
    ap.add_argument("--list", action="store_true", help="print the policy names and exit")
    ap.add_argument("-q", action="store_true", help="only print the summary table")
    a = ap.parse_args()

    if a.list:
        for name in POLICIES:
            print(name)
        return

    base = Params()
    for kv in a.set:
        k, v = kv.split("=", 1)
        if not hasattr(base, k):
            ap.error(f"unknown Params field {k!r}")
        cur = getattr(base, k)
        base = replace(base, **{k: type(cur)(ast.literal_eval(v))})

    fracs = [float(x) for x in a.fracs.split(",")]
    if a.policies:
        names = a.policies.split(",")
        unknown = [n for n in names if n not in POLICIES]
        if unknown:
            ap.error(f"unknown policies {unknown}; see --list")
    elif a.tier == "disk":
        names = [n for n in POLICIES if n != "cash-RAM-now"]
    else:
        names = ["cash-RAM-now", "LRU", "S3-FIFO", "ARC", "SLRU", "GDSF", "CostLRU",
                 "Supersede+LRU", "Supersede+CostLRU", "LiveGC+CostLRU"]
    results = {n: {f: [] for f in fracs} for n in names}
    hits = {n: {f: [] for f in fracs} for n in names}
    meta = []
    for seed in range(a.seeds):
        prm = replace(base, seed=seed)
        total, lb, eng0 = live_bytes(prm)
        if a.tier == "ram":
            lb = eng0.ram_live
        if a.tier == "disk":
            ram_cap = int(eng0.ram_live * a.ram_cap)
            t_inf = simulate(prm, Unlimited(), P.CashRAM(ram_cap)).time
            t_abs = simulate(prm, NoCache(), P.CashRAM(ram_cap)).time
        else:
            t_inf = simulate(prm, Unlimited(), Unlimited()).time
            t_abs = simulate(prm, Unlimited(), None).time
        meta.append(dict(seed=seed, footprint=total, live=lb, t_inf=t_inf, t_abs=t_abs))
        print(f"seed {seed}: disk footprint {total/1e9:.2f} GB, live {lb/1e9:.3f} GB "
              f"({total/max(1,lb):.0f}x), T_unlimited {t_inf/3600:.2f} h, "
              f"T_absent {t_abs/3600:.2f} h", file=sys.stderr)
        for f in fracs:
            cap = int(lb * f)
            for n in names:
                t0 = time.time()
                pol = POLICIES[n](cap)
                if a.tier == "disk":
                    eng = simulate(prm, pol, P.CashRAM(ram_cap), live_hints=n in NEEDS_LIVE,
                                   touch_hints=n in NEEDS_TOUCH, liveset_hints=n in NEEDS_LIVESET)
                    h = eng.disk_hits
                else:
                    eng = simulate(prm, Unlimited(), pol, live_hints=n in NEEDS_LIVE)
                    h = eng.ram_hits
                lost = (eng.time - t_inf) / max(1e-9, t_abs - t_inf)
                results[n][f].append(lost)
                hits[n][f].append(h / max(1, eng.requests))
                if not a.q:
                    print(f"  cap={f:>5}x  {n:<16} lost={lost*100:6.1f}%  "
                          f"tier-hits={h/max(1,eng.requests)*100:5.1f}%  ({time.time()-t0:.1f}s)",
                          file=sys.stderr)
    print()
    print(f"[{a.tier} tier] lost savings, mean over {a.seeds} seeds (lower is better); "
          f"cap = multiple of live-set bytes")
    print(f"{'policy':<18}" + "".join(f"{str(f)+'x':>9}" for f in fracs))
    for n in names:
        row = f"{n:<18}"
        for f in fracs:
            v = results[n][f]
            row += f"{statistics.mean(v)*100:8.1f}%"
        print(row)
    if a.tier == "disk":
        fp = statistics.mean(m["footprint"] / max(1, m["live"]) for m in meta)
        print(f"(unbounded disk footprint = {fp:.0f}x live set on average)")
    if a.out:
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out, "w") as fh:
            json.dump(dict(tier=a.tier, fracs=fracs, meta=meta,
                           results={n: {str(f): v for f, v in r.items()} for n, r in results.items()},
                           hits={n: {str(f): v for f, v in r.items()} for n, r in hits.items()}),
                      fh, indent=1)


if __name__ == "__main__":
    main()
