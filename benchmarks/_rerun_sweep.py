"""Re-run the notebook benchmark sweep on the current code.

Iterates every notebook in ref_notebooks.txt across off/cold/warm. Tolerates a
notebook that cannot run (missing data, missing optional dep) and reports it
rather than aborting the sweep, so one unavailable dataset does not cost the
other nine workloads.

Usage:
    python benchmarks/_rerun_sweep.py <results-dir> [--repeats N]
"""
from __future__ import annotations

import argparse
import os
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
DRIVER = REPO / "benchmarks" / "bench_notebook_overhead.py"
LIST = REPO / "benchmarks" / "ref_notebooks.txt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--timeout", type=int, default=900)
    args = ap.parse_args()

    out = REPO / args.results_dir
    out.mkdir(parents=True, exist_ok=True)
    notebooks = [ln.strip() for ln in LIST.read_text().splitlines() if ln.strip()]

    failures: list[str] = []
    for nb in notebooks:
        if not (REPO / nb).exists():
            print(f"SKIP  {nb} (not found)", flush=True)
            failures.append(f"{nb}: not found")
            continue
        for mode in ("off", "cold", "warm-session", "warm-restart"):
            t0 = time.perf_counter()
            # Deliberately NOT passing --cache-root: the driver defaults to
            # <results>/_caches/<notebook stem>, one cache per notebook. An
            # earlier version of this script passed a single shared root, which
            # put all ten workloads in one LRU where they evicted each other —
            # warm runs then restored almost nothing and it looked like cash
            # was failing to reuse anything.
            cmd = [
                sys.executable, str(DRIVER), nb, "--mode", mode,
                "--repeats", str(args.repeats),
                "--results-dir", str(out),
            ]
            # UTF-8 for the child: on Windows it would otherwise inherit
            # cp1252 and every notebook that prints a non-ASCII character
            # would die mid-cell. The driver also reconfigures its own stdio,
            # but setting it here covers anything it spawns in turn.
            env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
            try:
                # encoding/errors, not bare text=True: PYTHONIOENCODING makes
                # the CHILD write UTF-8, but the parent still decodes with the
                # locale codec, so on a Windows console any non-ASCII byte in a
                # notebook's output raises UnicodeDecodeError inside
                # communicate()'s reader thread. That thread dies, the
                # exception is printed by the threading excepthook rather than
                # raised here, and `p.stdout` comes back empty -- which is
                # survivable for a run that succeeds and silently loses the
                # error tail for one that fails.
                p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                                   encoding="utf-8", errors="replace",
                                   timeout=args.timeout, env=env)
            except subprocess.TimeoutExpired:
                print(f"TIMEOUT {nb} [{mode}]", flush=True)
                failures.append(f"{nb} [{mode}]: timeout")
                continue
            dt = time.perf_counter() - t0
            if p.returncode != 0:
                tail = (p.stderr or p.stdout).strip().splitlines()[-1:] or ["?"]
                print(f"FAIL  {nb} [{mode}] {dt:6.1f}s :: {tail[0][:110]}", flush=True)
                failures.append(f"{nb} [{mode}]: {tail[0][:160]}")
            else:
                print(f"ok    {nb} [{mode}] {dt:6.1f}s", flush=True)

    print("\n=== SWEEP DONE ===", flush=True)
    if failures:
        print(f"{len(failures)} failure(s):", flush=True)
        for f in failures:
            print("   ", f, flush=True)
    else:
        print("all runs succeeded", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
