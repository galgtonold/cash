"""Reference-notebook suite harness for size-estimator before/after.

Runs ``benchmarks/bench_notebook_overhead.py`` once per (notebook, mode)
across the 10-notebook reference suite. Output goes to a labelled
subdirectory of ``benchmarks/results/size_estimator_eval/`` so the
diff tool can compare two runs side-by-side.

Usage:
    python benchmarks/eval_size_estimator.py --label before
    python benchmarks/eval_size_estimator.py --label after  --repeats 3
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BENCH_SCRIPT = REPO_ROOT / "benchmarks" / "bench_notebook_overhead.py"
NOTEBOOK_LIST = REPO_ROOT / "benchmarks" / "ref_notebooks.txt"
OUT_ROOT = REPO_ROOT / "benchmarks" / "results" / "size_estimator_eval"


def _read_notebook_list(path: Path) -> list[Path]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [REPO_ROOT / line.strip() for line in lines
            if line.strip() and not line.startswith("#")]


def _run_one(notebook: Path, mode: str, results_dir: Path, repeats: int,
             cache_root: Path) -> int:
    """Invoke bench_notebook_overhead.py for one (notebook, mode). Returns exit code."""
    cmd = [
        sys.executable, str(BENCH_SCRIPT), str(notebook),
        "--mode", mode,
        "--repeats", str(repeats),
        "--results-dir", str(results_dir),
        "--cache-root", str(cache_root / notebook.stem),
        "--subprocess-per-repeat",
    ]
    print(f"  -> {notebook.name} [{mode}]")
    proc = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return proc.returncode


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Size-estimator reference-suite harness")
    p.add_argument("--label", required=True, choices=["before", "after"],
                   help="Output subdirectory label.")
    p.add_argument("--repeats", type=int, default=3,
                   help="Repeats per (notebook, mode). Repeat 0 is warmup. Default 3.")
    p.add_argument("--notebook-list", type=Path, default=NOTEBOOK_LIST,
                   help="Path to a file with one notebook path per line.")
    p.add_argument("--modes", nargs="+", default=["cold", "warm"],
                   help="Modes to run. Default: cold warm.")
    p.add_argument("--continue-on-error", action="store_true",
                   help="Continue running remaining notebooks if one fails.")
    args = p.parse_args(argv)

    out_dir = OUT_ROOT / args.label
    cache_root = OUT_ROOT / f"_caches_{args.label}"
    if out_dir.exists():
        shutil.rmtree(out_dir, ignore_errors=True)
    if cache_root.exists():
        shutil.rmtree(cache_root, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)

    notebooks = _read_notebook_list(args.notebook_list)
    print(f"Running {len(notebooks)} notebooks x {len(args.modes)} modes x "
          f"{args.repeats} repeats -> {out_dir}")

    n_failures = 0
    for notebook in notebooks:
        if not notebook.exists():
            print(f"  !! missing: {notebook}")
            n_failures += 1
            if not args.continue_on_error:
                return 2
            continue
        for mode in args.modes:
            rc = _run_one(notebook, mode, out_dir, args.repeats, cache_root)
            if rc != 0:
                print(f"  !! exit {rc} for {notebook.name} [{mode}]")
                n_failures += 1
                if not args.continue_on_error:
                    return rc

    print(f"\nWrote: {out_dir}")
    if n_failures:
        print(f"  with {n_failures} failures")
    return 0 if n_failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
