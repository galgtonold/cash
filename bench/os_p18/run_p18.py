"""Time p18-clinical's first run, plain vs cash, alternating rounds.

    python run_p18.py --venv VENV --out OUTDIR [--rounds 5] [--malloc-rounds 2]

Works on Windows and Linux (WSL). Writes OUTDIR/progress.txt after every run,
OUTDIR/results.jsonl (one line per run) and OUTDIR/SUMMARY.md at the end.
Runs already in results.jsonl are skipped, so a restart resumes.
--malloc-rounds adds plain and cash runs with glibc's mmap/trim thresholds
pinned (Linux only).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics as st
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PINS = {k: "1" for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS")}
MALLOC = {"MALLOC_MMAP_THRESHOLD_": "268435456", "MALLOC_TRIM_THRESHOLD_": "536870912"}
FREE_STOP_GB = 8.0


def vpy(venv):
    return os.path.join(venv, "Scripts", "python.exe") if os.name == "nt" else os.path.join(venv, "bin", "python")


def free_gb():
    try:
        import psutil

        return psutil.virtual_memory().available / 2**30
    except Exception:
        return None


def plan(rounds, malloc_rounds):
    out = []
    for r in range(1, rounds + 1):
        order = ("plain", "cash") if r % 2 else ("cash", "plain")
        out += [(f"r{r}", m, False) for m in order]
    if os.name != "nt":
        for r in range(1, malloc_rounds + 1):
            order = ("plain", "cash") if r % 2 else ("cash", "plain")
            out += [(f"m{r}", m, True) for m in order]
    return out


def one(venv, out, tag, mode, pinned):
    jl = os.path.join(out, f"{tag}.{mode}.jsonl")
    env = dict(os.environ, **PINS, CS_STATS="0", R34_VENV=venv)
    if pinned:
        env.update(MALLOC)
    t0 = time.time()
    with open(jl + ".log", "w", encoding="utf-8") as log:
        rc = subprocess.call(
            [
                vpy(venv),
                os.path.join(HERE, "harness", "replay.py"),
                os.path.join(HERE, "p18", "trace_cold.json"),
                os.path.join(HERE, "p18", "project"),
                mode,
                jl,
            ],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    runs = [json.loads(line) for line in open(jl, encoding="utf-8") if line.strip()] if os.path.exists(jl) else []
    runs = [r for r in runs if r.get("op") == "run"]
    return {
        "tag": tag,
        "mode": mode,
        "malloc_pinned": pinned,
        "rc": rc,
        "run_wall": round(sum(r.get("wall", 0) for r in runs), 2),
        "boot_wall": round(sum(r.get("wall", 0) for r in runs if r.get("cell") == "boot"), 2),
        "errors": sum(r.get("status") != "ok" for r in runs),
        "runs": len(runs),
        "elapsed": round(time.time() - t0, 1),
        "free_gb_after": free_gb(),
        "load": os.getloadavg() if hasattr(os, "getloadavg") else None,
    }


def summary(out, rows):
    lines = [f"# p18 first run on {platform.system()} {platform.release()} ({platform.python_version()})", ""]
    lines.append("| set | mode | run_wall median (all) | boot cell median (min-max) | errors |")
    lines.append("|---|---|---|---|---|")
    for pinned in (False, True):
        for mode in ("plain", "cash"):
            rs = [r for r in rows if r["mode"] == mode and r["malloc_pinned"] == pinned]
            if not rs:
                continue
            w = [r["run_wall"] for r in rs]
            b = [r["boot_wall"] for r in rs]
            lines.append(
                f"| {'malloc pinned' if pinned else 'default'} | {mode} | {st.median(w):.1f} ({', '.join(f'{x:.1f}' for x in w)}) "
                f"| {st.median(b):.1f} ({min(b):.1f}-{max(b):.1f}) | {sum(r['errors'] for r in rs)} |"
            )
    for pinned in (False, True):
        p = [r["run_wall"] for r in rows if r["mode"] == "plain" and r["malloc_pinned"] == pinned]
        c = [r["run_wall"] for r in rows if r["mode"] == "cash" and r["malloc_pinned"] == pinned]
        if p and c:
            lines.append(
                f"\nratio cash/plain ({'malloc pinned' if pinned else 'default'}): {st.median(c) / st.median(p):.3f}"
            )
    open(os.path.join(out, "SUMMARY.md"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--venv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rounds", type=int, default=5)
    ap.add_argument("--malloc-rounds", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    res = os.path.join(a.out, "results.jsonl")
    rows = [json.loads(line) for line in open(res, encoding="utf-8")] if os.path.exists(res) else []
    done = {(r["tag"], r["mode"]) for r in rows}
    todo = plan(a.rounds, a.malloc_rounds)
    for tag, mode, pinned in todo:
        if (tag, mode) in done:
            continue
        f = free_gb()
        if f is not None and f < FREE_STOP_GB:
            msg = f"STOPPED: only {f:.1f} GB free before {tag} {mode}"
            open(os.path.join(a.out, "progress.txt"), "a", encoding="utf-8").write(msg + "\n")
            print(msg)
            sys.exit(2)
        row = one(a.venv, a.out, tag, mode, pinned)
        rows.append(row)
        open(res, "a", encoding="utf-8").write(json.dumps(row) + "\n")
        line = f"{time.strftime('%H:%M:%S')} {len(rows)}/{len(todo)} {json.dumps(row)}"
        open(os.path.join(a.out, "progress.txt"), "a", encoding="utf-8").write(line + "\n")
        print(line, flush=True)
    summary(a.out, rows)
    open(os.path.join(a.out, "DONE"), "w", encoding="utf-8").write("done\n")


if __name__ == "__main__":
    main()
