"""Notebook overhead benchmark CLI.

Run a notebook in cash-off / cash-cold / cash-warm modes and write
per-cell timings + an optional pyinstrument profile to ``--results-dir``.

Usage:
    python benchmarks/bench_notebook_overhead.py <notebook> --mode {off,cold,warm}
        [--profile] [--repeats N] [--results-dir DIR] [--cache-root DIR]
"""
from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Make the benchmarks package importable when invoked as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks._overhead_driver import new_cash_session, run_notebook
from benchmarks._overhead_io import load_code_cells
from benchmarks._overhead_results import RunResult, write_results

# The two warm paths a notebook user actually has. They differ in one thing:
# whether the RAM tier survives between the priming run and the measured one.
#
#   warm-session   same kernel, run the cells again. RAM + disk are both live.
#                  This is the everyday iteration loop.
#   warm-restart   kernel restarted (or a new process). Only what the cost
#                  model wrote to DISK can come back.
#
# These were one mode called "warm", which built a fresh Cash for the measured
# pass and so behaved as warm-restart while being read as though it covered
# both. Values the cost model parks in RAM — most statements under ~0.1s of
# compute — were scored as misses, which is why several reference notebooks
# reported zero restores while caching perfectly well.
WARM_MODES = ("warm-session", "warm-restart")
_MODE_ALIASES = {"warm": "warm-restart"}


def _dir_size_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    total = 0
    for p in path.rglob("*"):
        if p.is_file():
            try:
                total += p.stat().st_size
            except OSError:
                pass
    return total


def _cash_version() -> str:
    try:
        from cash import __version__
        return __version__
    except Exception:  # noqa: BLE001
        return "unknown"


def _run_once(
    notebook_path: Path,
    mode: str,
    repeat: int,
    cache_dir: Path | None,
    profile_path: Path | None,
    session=None,
) -> RunResult:
    cells = load_code_cells(notebook_path)

    profiler = None
    if profile_path is not None:
        from pyinstrument import Profiler
        profiler = Profiler(interval=0.001)
        profiler.start()

    t0 = time.perf_counter()
    timings = run_notebook(
        cells,
        cash_enabled=(mode != "off"),
        cache_dir=cache_dir,
        session=session,
    )
    total = time.perf_counter() - t0

    if profiler is not None:
        profiler.stop()
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text(profiler.output_html(), encoding="utf-8")

    return RunResult(
        notebook=str(notebook_path),
        mode=mode,
        repeat=repeat,
        python_version=sys.version.split()[0],
        cash_version=_cash_version(),
        platform=platform.platform(),
        cells=timings,
        total_wall_seconds=total,
        cache_dir_bytes=_dir_size_bytes(cache_dir) if cache_dir else 0,
    )


def _force_utf8_stdio() -> None:
    """Make this process's stdio UTF-8, like a real Jupyter kernel.

    Under a Windows console the benchmark child inherits cp1252, and any
    notebook cell that prints a non-ASCII character dies with
    UnicodeEncodeError partway through. The cell still returns a timing, so
    the damage surfaces as *missing statement metrics* rather than as an
    encoding error -- 09_yelp_reviews lost its data-generation cell (and with
    it every downstream cell, to NameError), cfd lost three cells, and
    10_us_flights produced no metrics at all. It read exactly like a caching
    defect and was chased as one.

    A kernel is UTF-8, so matching it is also the more faithful measurement.
    ``errors='replace'`` because a benchmark must never die over a glyph.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):  # not a reconfigurable stream
            pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdio()
    p = argparse.ArgumentParser(description="Notebook overhead benchmark")
    p.add_argument("notebook", type=Path)
    p.add_argument("--mode", required=True,
                   choices=["off", "cold", "warm-session", "warm-restart",
                            "warm"],
                   help="'warm' is a deprecated alias for warm-restart; it "
                        "only ever measured disk-tier restores.")
    p.add_argument("--profile", action="store_true",
                   help="Wrap the run in pyinstrument and emit HTML")
    p.add_argument("--repeats", type=int, default=3,
                   help="Number of repeats; the first is reported but typical "
                        "analysis discards it as warmup (default: 3)")
    p.add_argument("--results-dir", type=Path,
                   default=Path("benchmarks/results"))
    p.add_argument("--cache-root", type=Path,
                   help="Parent dir for per-repeat cache dirs (cold/warm only)")
    p.add_argument("--single-repeat", type=int, default=None,
                   help="Internal: run only this single repeat index and exit "
                        "(used by --subprocess-per-repeat to fork the workload).")
    p.add_argument("--subprocess-per-repeat", action="store_true",
                   help="Fork a fresh subprocess per repeat. Necessary for "
                        "notebooks that use cash machinery internally (e.g. "
                        "%%load_ext cash), because cash's monkey-patches "
                        "and module-level state leak across repeats in one "
                        "process and contaminate cold-mode measurements.")
    args = p.parse_args(argv)

    if args.mode in _MODE_ALIASES:
        resolved = _MODE_ALIASES[args.mode]
        print(f"note: --mode {args.mode} is deprecated; running "
              f"{resolved}. It measures kernel-restart survival (disk tier "
              f"only). For the same-kernel iteration loop use warm-session.")
        args.mode = resolved

    if args.mode != "off" and args.cache_root is None:
        args.cache_root = args.results_dir / "_caches" / args.notebook.stem

    stem = args.notebook.stem

    # --- Parent orchestrator: subprocess-per-repeat mode ------------------
    # We re-exec ourselves once per repeat with --single-repeat <N>. This
    # is necessary for notebooks that load cash inside the notebook (e.g.
    # %load_ext cash): cash's FileAccessTracker monkey-patches and
    # module-level state leak across repeats within one Python process,
    # contaminating cold/warm measurements. A fresh subprocess guarantees
    # a clean Python state per repeat.
    #
    # The parent decides what CASH_CACHE_DIR each child uses by setting
    # it in env BEFORE spawning the subprocess. The child inherits it.
    # Children with --single-repeat skip the env reset that the parent
    # otherwise does.
    if args.subprocess_per_repeat and args.single_repeat is None:
        global_cash_root = args.results_dir / "_global_cash" / stem
        if global_cash_root.exists():
            shutil.rmtree(global_cash_root, ignore_errors=True)
        global_cash_root.mkdir(parents=True, exist_ok=True)

        # Warm mode: populate a shared cache dir once, then each repeat
        # uses it. The populate run uses cold mode so cash actually
        # writes entries; its CASH_CACHE_DIR is the same warm_shared so
        # any global cash created in-notebook also populates it.
        # Only warm-restart primes from a separate process. warm-session must
        # prime inside the process that measures — a RAM tier cannot cross a
        # process boundary — so its child does its own priming below.
        if args.mode == "warm-restart":
            warm_shared = global_cash_root / "warm-shared"
            if warm_shared.exists():
                shutil.rmtree(warm_shared, ignore_errors=True)
            warm_shared.mkdir(parents=True, exist_ok=True)
            env = {**os.environ, "CASH_CACHE_DIR": str(warm_shared)}
            subprocess.run(
                [sys.executable, __file__, str(args.notebook),
                 "--mode", "cold", "--repeats", "1",
                 "--results-dir", str(args.results_dir / "_warm_prepop"),
                 "--cache-root", str(warm_shared),
                 "--single-repeat", "0"],
                check=True, env=env,
            )

        for repeat in range(args.repeats):
            if args.mode == "cold":
                per_repeat_dir = global_cash_root / f"cold-{repeat}"
                if per_repeat_dir.exists():
                    shutil.rmtree(per_repeat_dir, ignore_errors=True)
                per_repeat_dir.mkdir(parents=True, exist_ok=True)
                cash_dir = per_repeat_dir
            elif args.mode in WARM_MODES:
                cash_dir = global_cash_root / "warm-shared"
            else:  # off — isolate per repeat so any in-notebook cash use
                   # doesn't accumulate state
                per_repeat_dir = global_cash_root / f"off-{repeat}"
                if per_repeat_dir.exists():
                    shutil.rmtree(per_repeat_dir, ignore_errors=True)
                per_repeat_dir.mkdir(parents=True, exist_ok=True)
                cash_dir = per_repeat_dir

            env = {**os.environ, "CASH_CACHE_DIR": str(cash_dir)}
            subprocess.run(
                [sys.executable, __file__, str(args.notebook),
                 "--mode", args.mode, "--repeats", str(args.repeats),
                 "--results-dir", str(args.results_dir),
                 "--cache-root", str(args.cache_root),
                 "--single-repeat", str(repeat)],
                check=True, env=env,
            )
        return 0

    # --- Single-repeat / non-subprocess path -----------------------------
    # If we're a single-repeat subprocess child, CASH_CACHE_DIR is already
    # set in our env by the parent — don't override it. If we're a fully
    # standalone in-process run (no --subprocess-per-repeat), set up an
    # isolated CASH_CACHE_DIR like before.
    if args.single_repeat is None:
        global_cash_dir = args.results_dir / "_global_cash" / f"{stem}-{args.mode}"
        if global_cash_dir.exists():
            shutil.rmtree(global_cash_dir, ignore_errors=True)
        global_cash_dir.mkdir(parents=True, exist_ok=True)
        os.environ["CASH_CACHE_DIR"] = str(global_cash_dir)

    # --- Warm priming ----------------------------------------------------
    # Populate the cache once, then measure re-runs against it.
    #
    # warm-restart primes with its own throwaway Cash and each measured repeat
    # builds another, so only values that reached disk survive the handover.
    # warm-session threads ONE Cash through the priming run and every repeat,
    # keeping the RAM tier alive — and therefore must prime in this process,
    # including when the parent forked us with --single-repeat.
    warm_session = None
    if args.mode == "warm-session":
        warm_dir = args.cache_root / "warm-shared"
        if args.single_repeat is None and warm_dir.exists():
            shutil.rmtree(warm_dir, ignore_errors=True)
        warm_dir.mkdir(parents=True, exist_ok=True)
        warm_session = new_cash_session(warm_dir)
        _run_once(args.notebook, "cold", -1, warm_dir, profile_path=None,
                  session=warm_session)
    elif args.mode == "warm-restart" and args.single_repeat is None:
        warm_dir = args.cache_root / "warm-shared"
        if warm_dir.exists():
            shutil.rmtree(warm_dir, ignore_errors=True)
        warm_dir.mkdir(parents=True, exist_ok=True)
        _run_once(args.notebook, "cold", -1, warm_dir, profile_path=None)

    repeat_range = (
        [args.single_repeat]
        if args.single_repeat is not None
        else range(args.repeats)
    )
    for repeat in repeat_range:
        if args.mode == "cold":
            cache_dir = args.cache_root / f"repeat-{repeat}"
            if cache_dir.exists():
                shutil.rmtree(cache_dir, ignore_errors=True)
            cache_dir.mkdir(parents=True, exist_ok=True)
            # In the non-subprocess in-process path we also wipe the
            # global-cash dir between cold repeats so every cold repeat
            # is truly cold. In the subprocess-per-repeat path the
            # parent handles CASH_CACHE_DIR per repeat, so we skip this.
            if args.single_repeat is None:
                for entry in global_cash_dir.iterdir():
                    if entry.is_file():
                        entry.unlink()
                    elif entry.is_dir():
                        shutil.rmtree(entry, ignore_errors=True)
        elif args.mode in WARM_MODES:
            cache_dir = args.cache_root / "warm-shared"
        else:
            cache_dir = None

        profile_path = None
        if args.profile and repeat == args.repeats - 1:
            # Only profile the last repeat (steady state); profiling adds
            # overhead that distorts the wall-clock comparison.
            profile_path = args.results_dir / f"{stem}-{args.mode}-profile.html"

        result = _run_once(args.notebook, args.mode, repeat, cache_dir,
                           profile_path, session=warm_session)
        out = args.results_dir / f"{stem}-{args.mode}-{repeat}.json"
        write_results(out, result)
        print(f"[{args.mode}] repeat={repeat} cells={len(result.cells)} "
              f"wall={result.total_wall_seconds*1000:.1f}ms -> {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
