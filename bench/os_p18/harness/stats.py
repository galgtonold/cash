#!/usr/bin/env python3
"""Per-item stats for one correctness-set run directory (speed and correctness in one place).

usage: stats.py RUNS_DIR

Reads RUNS_DIR/speed.json (speed.py's per-step plain / cash / served / free) and the
cash run speed.py timed (normally cash2), and writes RUNS_DIR/stats.json and stats.md:

* per cell run: plain s, cash s, served from cache or not, and when not, why
  (cash's own decision record per statement: skipped_reason / uncacheable_reasons /
  miss_reason / guard_cause; where cash gives none, the harness's inference, marked
  "(inferred)": first run, edited code, or same code run again)
* the session split into first runs, repeats (same kernel) and runs after a restart:
  P (plain), C (cash), Z1 (plain - served: cash with no overhead), Z2 (plain - free:
  a perfect cache) for each
* cash's work counts per cell and in total (cs_stats.py: bytes hashed, frame/value
  copies, pickles, referrer scans, source parses, store/restore seconds) and the
  counters' own cost
* warnings raised (category and code, count) and cache size on disk at the end
* where time is saved, where it is lost (overhead), and the top miss reasons by seconds
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict

CLASSES = ("first", "repeat", "after_restart")


def load_jsonl(path):
    meta, recs, end = None, {}, None
    if not os.path.exists(path):
        return meta, recs, end
    for line in open(path, encoding="utf-8"):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r.get("op") == "meta":
            meta = r
        elif r.get("op") == "end":
            end = r
        else:
            recs[r["i"]] = r
    return meta, recs, end


_NUM = re.compile(r"\d+(?:\.\d+)?(?:e-?\d+)?")
_QUOTED = re.compile(r"'[^']*'|\"[^\"]*\"|`[^`]*`")


def bucket(reason: str) -> str:
    """A reason with names, numbers and quoted parts taken out, so like reasons add up."""
    r = str(reason).strip().splitlines()[0] if str(reason).strip() else ""
    r = _QUOTED.sub("'…'", r)
    r = _NUM.sub("N", r)
    return r[:110]


def stmt_reason(st: dict) -> tuple[str, str] | None:
    """(source, reason) from cash's own record for one statement that was not served."""
    for k in ("skipped_reason", "miss_reason", "guard_cause"):
        v = st.get(k)
        if v:
            return k, bucket(v)
    u = st.get("uncacheable_reasons")
    if u:
        return "uncacheable", bucket(u[0] if isinstance(u, list) else u)
    return None


def storage_of(rec: dict) -> set:
    out = set()
    for st in ((rec.get("cash") or {}).get("raw") or []):
        v = st.get("storage")
        out.update(v if isinstance(v, list) else [v] if v else [])
    return out


def cell_reason(rec: dict, history: dict, cls: str, stored: dict | None = None) -> tuple[str, list]:
    """Why a cell run was not (fully) served: cash's reasons, else an inferred one."""
    cash = rec.get("cash") or {}
    raw = cash.get("raw") or []
    found = []
    for st in raw:
        if st.get("is_upstream"):
            continue
        status = str(st.get("status") or "").upper()
        if status in ("RESTORED", "SKIPPED", "CACHED"):
            continue
        r = stmt_reason(st)
        if r:
            found.append((float(st.get("execution_time") or 0.0), r[0] + ": " + r[1]))
    found.sort(reverse=True)
    if found:
        return found[0][1], [f for _, f in found[:4]]
    prev = history.get(rec.get("cell"))
    if prev is None:
        why = "first run (inferred)"
    elif prev != rec.get("src"):
        why = "edited code (inferred)"
    elif cls == "after_restart":
        before = (stored or {}).get(rec.get("cell"))
        if before is None or not before:
            why = "after restart: not stored on the earlier run (inferred)"
        elif before <= {"RAM"}:
            why = "after restart: earlier result was kept in memory only (inferred)"
        else:
            why = "after restart: stored on disk earlier, not served (inferred)"
    else:
        why = "same code run again, not served (inferred)"
    return why, []


def main():
    rundir = sys.argv[1]
    sp_path = os.path.join(rundir, "speed.json")
    if not os.path.exists(sp_path):
        print(f"stats: no speed.json in {rundir}", file=sys.stderr)
        sys.exit(1)
    sp = json.load(open(sp_path, encoding="utf-8"))
    crun = sp.get("cash_run") or "cash2"
    prun = sp.get("plain_run") or "plain1"
    _cm, crecs, cend = load_jsonl(os.path.join(rundir, crun + ".jsonl"))
    _pm, precs, _pe = load_jsonl(os.path.join(rundir, prun + ".jsonl"))

    # classify each run step from the trace order (same in plain and cash)
    seen, since_restart, history = set(), set(), {}
    cls_of = {}
    for i in sorted(precs):
        r = precs[i]
        if r.get("op") == "restart" or r.get("forced_restart"):
            since_restart = set()
            if r.get("op") == "restart":
                continue
        if r.get("op") != "run":
            continue
        c = r.get("cell")
        if c not in seen:
            cls_of[i] = "first"
        elif c not in since_restart:
            cls_of[i] = "after_restart"
        else:
            cls_of[i] = "repeat"
        seen.add(c)
        since_restart.add(c)
        if r.get("forced_restart"):
            since_restart = set()

    split = {k: {"n": 0, "P": 0.0, "C": 0.0, "served": 0.0, "free": 0.0, "overhead": 0.0} for k in CLASSES + ("other",)}
    cells, miss_by_reason, over_by_cell, stored = [], defaultdict(lambda: [0, 0.0]), [], {}
    work_tot, own_tot, warn_tot = defaultdict(lambda: [0, 0.0, 0]), 0.0, defaultdict(int)
    for row in sp.get("steps", []):
        i = row["i"]
        cls = cls_of.get(i, "other")
        s = split[cls]
        s["n"] += 1
        s["P"] += row.get("plain") or 0.0
        s["C"] += row.get("cash") or 0.0
        s["served"] += row.get("served") or 0.0
        s["free"] += row.get("free") or 0.0
        s["overhead"] += row.get("overhead") or 0.0
        cr = crecs.get(i, {})
        work = ((cr.get("cash") or {}).get("work") or {}) if row["op"] == "run" else {}
        for k, v in work.items():
            if k == "_own":
                own_tot += v
            elif isinstance(v, list):
                w = work_tot[k]
                w[0] += v[0]
                w[1] += v[1]
                w[2] += v[2]
        for k, n in (cr.get("warnings") or {}).items():
            warn_tot[k] += n
        if row["op"] != "run":
            continue
        pr = precs.get(i, {})
        served = (row.get("served") or 0.0) > 0.005
        missed = row.get("missed") or 0.0
        why, why_all = ("", [])
        if missed > 0.005 or not served:
            why, why_all = cell_reason({**cr, "cell": row.get("cell"), "src": pr.get("src")}, history, cls, stored)
        if missed > 0.005:
            m = miss_by_reason[why]
            m[0] += 1
            m[1] += missed
        history[row.get("cell")] = pr.get("src")
        stored[row.get("cell")] = storage_of(cr)
        cells.append({
            "i": i, "cell": row.get("cell"), "class": cls, "plain": row.get("plain"), "cash": row.get("cash"),
            "served": row.get("served"), "free": row.get("free"), "missed": round(missed, 4),
            "overhead": row.get("overhead"), "is_served": served, "why_not": why or None, "why_all": why_all or None,
            "work": {k: v for k, v in work.items() if k != "_own"} or None,
            "warnings": cr.get("warnings") or None, "status": cr.get("status"),
        })
        if row.get("overhead") is not None:
            over_by_cell.append((row["overhead"], i, row.get("cell"), row.get("overhead_excerpt") or ""))

    for s in split.values():
        s["Z1"] = s["P"] - s["served"]
        s["Z2"] = s["P"] - s["free"]
        for k in list(s):
            if isinstance(s[k], float):
                s[k] = round(s[k], 3)
    total = {k: round(sum(split[c][k] for c in split), 3) for k in ("n", "P", "C", "served", "free", "overhead", "Z1", "Z2")}
    over_by_cell.sort(reverse=True)
    saved_by_cell = sorted(((c["served"] or 0.0, c["i"], c["cell"]) for c in cells), reverse=True)
    res = {
        "rundir": os.path.abspath(rundir), "cash_run": crun, "plain_run": prun,
        "split": split, "total": total,
        "top_miss_reasons": sorted(([k, v[0], round(v[1], 3)] for k, v in miss_by_reason.items()), key=lambda x: -x[2]),
        "top_overhead": [[round(o, 3), i, c, e[:100]] for o, i, c, e in over_by_cell[:10]],
        "top_saved": [[round(s, 3), i, c] for s, i, c in saved_by_cell[:10] if s > 0],
        "work": {k: [v[0], round(v[1], 3), v[2]] for k, v in sorted(work_tot.items())},
        "recording_cost_s": round(own_tot, 4),
        "warnings": dict(sorted(warn_tot.items(), key=lambda x: -x[1])),
        "cache_bytes": (cend or {}).get("cache_bytes"), "cache_files": (cend or {}).get("cache_files"),
        "cells": cells,
    }
    json.dump(res, open(os.path.join(rundir, "stats.json"), "w", encoding="utf-8"), indent=1, default=str)

    f = lambda x: "-" if x is None else f"{x:.2f}"  # noqa: E731
    md = [f"# stats: {os.path.basename(os.path.abspath(rundir.rstrip('/')))} ({crun} vs {prun})", "",
          "## Time split", "", "| runs | n | plain s | cash s | Z1 s | Z2 s | served s | overhead s |", "|---|---|---|---|---|---|---|---|"]
    for k in CLASSES + ("other",):
        s = split[k]
        if s["n"]:
            md.append(f"| {k} | {s['n']} | {f(s['P'])} | {f(s['C'])} | {f(s['Z1'])} | {f(s['Z2'])} | {f(s['served'])} | {f(s['overhead'])} |")
    t = total
    md.append(f"| **all** | {t['n']} | {f(t['P'])} | {f(t['C'])} | {f(t['Z1'])} | {f(t['Z2'])} | {f(t['served'])} | {f(t['overhead'])} |")
    md += ["", "Z1 = plain - served (cash with no overhead), Z2 = plain - free (a perfect cache). 'other' = kernel starts, scripts, file writes.",
           "", "## Top miss reasons (seconds a perfect cache would have saved)", "", "| reason | cells | s |", "|---|---|---|"]
    for k, n, s in res["top_miss_reasons"][:12]:
        md.append(f"| {k.replace('|', '/')} | {n} | {f(s)} |")
    md += ["", "## Where cash costs time (top overhead cells)", "", "| s | step | cell | statement |", "|---|---|---|---|"]
    for o, i, c, e in res["top_overhead"][:8]:
        md.append(f"| {f(o)} | {i} | {c} | {e.replace('|', '/')} |")
    md += ["", "## Where cash saves time", "", "| s | step | cell |", "|---|---|---|"]
    for s, i, c in res["top_saved"][:8]:
        md.append(f"| {f(s)} | {i} | {c} |")
    md += ["", "## Cash's work (all cells)", "", "| counter | calls | s | bytes |", "|---|---|---|---|"]
    for k, v in res["work"].items():
        md.append(f"| {k} | {v[0]} | {f(v[1])} | {v[2] or ''} |")
    md += ["", f"Recording cost (the counters' own time): {res['recording_cost_s']:.3f} s.",
           f"Cache on disk at the end: {res['cache_bytes']} bytes in {res['cache_files']} files." if res["cache_bytes"] is not None else "Cache size: not recorded.",
           "", "## Warnings", ""]
    md += [f"- {k}: {n}" for k, n in res["warnings"].items()] or ["none"]
    md += ["", "## Cells", "", "| step | cell | runs | plain s | cash s | served | missed s | why not served |", "|---|---|---|---|---|---|---|---|"]
    for c in cells:
        md.append(f"| {c['i']} | {c['cell']} | {c['class']} | {f(c['plain'])} | {f(c['cash'])} | {'yes' if c['is_served'] else 'no'} | {f(c['missed'])} | {(c['why_not'] or '').replace('|', '/')} |")
    open(os.path.join(rundir, "stats.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    print(f"stats: {rundir}/stats.md")


if __name__ == "__main__":
    main()
