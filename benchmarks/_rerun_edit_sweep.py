"""Run the edit benchmark across every reference notebook.

Sibling of ``_rerun_sweep.py``: same tolerate-and-report posture, same
subprocess-per-notebook isolation. One notebook that cannot run (missing
dataset, missing optional dep) is reported and skipped rather than aborting
the sweep.

A subprocess per notebook is not incidental. The edit benchmark drives an
in-process ``InteractiveShell`` and a global ``Cash`` singleton; running ten
notebooks in one process would let one notebook's shell state decide the
next one's cache statuses, which is the exact confound the numbers exist to
avoid.

Usage:
    python benchmarks/_rerun_edit_sweep.py <results-dir>
        [--max-sites N] [--session {restart,live}] [--timeout S]
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
DRIVER = REPO / "benchmarks" / "bench_notebook_edit.py"
LIST = REPO / "benchmarks" / "ref_notebooks.txt"


def _summarise(results_dir: pathlib.Path, session_mode: str) -> None:
    """Print the one table this whole sweep exists to produce."""
    rows = []
    for path in sorted(results_dir.glob(f"*-edit-{session_mode}.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        nulls = [s for s in report["scenarios"] if s["kind"] != "linked"]
        controls = [s for s in report["scenarios"] if s["kind"] == "linked"]
        rows.append((
            pathlib.Path(report["notebook"]).name,
            report["restorable_count"],
            sum(s["wasted_count"] for s in nulls),
            sum(s["wasted_seconds"] for s in nulls),
            len(nulls),
            all(c["control_sink_recomputed"] for c in controls) if controls else None,
        ))

    if not rows:
        print("\nno results to summarise")
        return

    print(f"\n=== edit sweep ({session_mode}) ===")
    print(f"{'notebook':38s} {'restorable':>10s} {'wasted':>8s} "
          f"{'wasted s':>9s} {'scenarios':>10s} {'control':>8s}")
    for name, restorable, wasted, secs, n, control_ok in rows:
        # A control that did not fire means this notebook's zeros are not
        # evidence of anything -- say so on the row rather than in a footnote.
        control = "-" if control_ok is None else ("ok" if control_ok else "BROKEN")
        print(f"{name[:38]:38s} {restorable:10d} {wasted:8d} "
              f"{secs:9.2f} {n:10d} {control:>8s}")
    print("\n'wasted' counts statements that restore when nothing is edited "
          "but recompute after an edit\nthat cannot have invalidated them. "
          "Zero is the target; a BROKEN control voids that row.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--max-sites", type=int, default=3)
    ap.add_argument("--session", dest="session_mode", default="restart",
                    choices=["restart", "live"])
    ap.add_argument("--timeout", type=int, default=1800)
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
        t0 = time.perf_counter()
        cmd = [
            sys.executable, str(DRIVER), nb,
            "--results-dir", str(out),
            "--max-sites", str(args.max_sites),
            "--session", args.session_mode,
            "--quiet",
        ]
        # UTF-8 for the child, for the same reason the overhead sweep does
        # it: a cp1252 console kills any cell that prints a non-ASCII glyph
        # partway through, and the damage shows up as missing metrics rather
        # than as an error. The parent needs telling separately -- without
        # encoding/errors here it decodes the child's UTF-8 with the locale
        # codec and loses the output to a dead reader thread.
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        try:
            p = subprocess.run(cmd, cwd=REPO, capture_output=True, text=True,
                               encoding="utf-8", errors="replace",
                               timeout=args.timeout, env=env)
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT {nb}", flush=True)
            failures.append(f"{nb}: timeout")
            continue
        dt = time.perf_counter() - t0
        if p.returncode != 0:
            tail = (p.stderr or p.stdout).strip().splitlines()[-1:] or ["?"]
            print(f"FAIL  {nb} {dt:6.1f}s :: {tail[0][:110]}", flush=True)
            failures.append(f"{nb}: {tail[0][:160]}")
        else:
            print(f"ok    {nb} {dt:6.1f}s", flush=True)

    print("\n=== EDIT SWEEP DONE ===", flush=True)
    if failures:
        print(f"{len(failures)} failure(s):", flush=True)
        for f in failures:
            print("   ", f, flush=True)
    _summarise(out, args.session_mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
