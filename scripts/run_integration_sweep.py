"""Run the notebook integration suite in chunks, keeping failure output.

The suite is ~840 files of real kernels executing real notebooks. Running it
in one pytest invocation HANGS at roughly 175 files under ``-n 16`` -- a
worker-death cascade -- while every sub-group of <=120 passes. So it is run
in chunks, and the chunk size is the point of this script rather than a
detail of it.

Two settings are load-bearing:

* ``--chunk 60``. Comfortably under the observed cliff, with headroom for a
  slower machine where the cascade starts earlier.
* ``-n 16``, not ``-n auto``. The suite is kernel-boot-throttle-bound, and
  oversubscribing both slows it and destabilises it. ``worksteal`` rather
  than ``loadscope``: loadscope hands a whole module to one worker, so a
  worker that dies takes its module with it.

Failure output is written to ``<results>/chunkN_failure.log`` and echoed.
That is not a nicety. The first version of this printed only the ``FAILED``
line, and the one real failure it caught -- an oracle test comparing cash's
answer against a no-cache run -- could not be inspected afterwards, which
cost an hour of inference that one traceback would have settled.

Usage:
    python scripts/run_integration_sweep.py [--chunk 60] [-n 16]
        [--results-dir DIR] [--only SUBSTRING]
"""
from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys
import time

REPO = pathlib.Path(__file__).resolve().parents[1]
SUITE = REPO / "tests" / "test_notebook_integration"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=60,
                    help="files per pytest invocation (default: 60)")
    ap.add_argument("-n", "--workers", default="16",
                    help="xdist workers; 'auto' is deliberately NOT the default")
    ap.add_argument("--results-dir", type=pathlib.Path,
                    default=REPO / "integration_sweep")
    ap.add_argument("--only", default="",
                    help="only files whose name contains this substring")
    ap.add_argument("--timeout", type=int, default=3600)
    args = ap.parse_args()

    files = sorted(p.as_posix() for p in SUITE.glob("test_*.py")
                   if args.only in p.name)
    if not files:
        print(f"no test files matched {args.only!r} in {SUITE}")
        return 1

    args.results_dir.mkdir(parents=True, exist_ok=True)
    chunks = [files[i:i + args.chunk] for i in range(0, len(files), args.chunk)]
    print(f"{len(files)} files in {len(chunks)} chunk(s) of <= {args.chunk}, "
          f"-n {args.workers}", flush=True)

    failures: list[tuple[int, str]] = []
    started = time.perf_counter()
    for i, chunk in enumerate(chunks, 1):
        t0 = time.perf_counter()
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", *chunk, "-q",
             "-n", args.workers, "--dist", "worksteal",
             "-p", "no:randomly", "-rf", "--tb=long"],
            cwd=REPO, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=args.timeout,
        )
        elapsed = time.perf_counter() - t0
        out = proc.stdout or ""
        tail = [ln for ln in out.strip().splitlines()
                if " in " in ln and any(w in ln for w in
                                        ("passed", "failed", "error"))]
        summary = tail[-1] if tail else "(no summary line)"

        if proc.returncode == 0:
            print(f"ok   chunk {i:2d}/{len(chunks)} {elapsed:6.1f}s  {summary}",
                  flush=True)
            continue

        log = args.results_dir / f"chunk{i}_failure.log"
        log.write_text(out + "\n===STDERR===\n" + (proc.stderr or ""),
                       encoding="utf-8")
        failures.append((i, summary))
        print(f"FAIL chunk {i:2d}/{len(chunks)} {elapsed:6.1f}s  {summary}",
              flush=True)
        for line in out.splitlines():
            if line.startswith(("FAILED", "ERROR")):
                print(f"     {line[:160]}", flush=True)
        print(f"     full output: {log}", flush=True)

    print(f"\n=== SWEEP DONE in {(time.perf_counter() - started) / 60:.1f} min ===",
          flush=True)
    if not failures:
        # Deliberately not "all green": this suite has no known-red baseline,
        # so "0 failed" is the only acceptable result and saying so plainly
        # keeps anyone from inventing one to subtract.
        print("0 failed. Any failure here is real -- there is no known-red set.")
        return 0

    print(f"{len(failures)} chunk(s) with failures:")
    for i, summary in failures:
        print(f"   chunk {i}: {summary}")
    print("\nAttribute before blaming your change: re-run the chunk alone, and "
          "if it reproduces, `git stash push -- src/` and run it again.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
