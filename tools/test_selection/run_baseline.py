"""Run the notebook integration suite once with per-test coverage.

Produces, under ``--out`` (default ``.testsel/``):

* ``results.json``  - outcome, duration and rerun count for every test
* ``cov/.coverage`` - combined coverage, each line labelled with the tests
  that ran it (coverage.py contexts)
* ``run_info.json`` - commit, platform, timings, pytest exit code
* ``pytest.log``    - full pytest output

then runs ``select_core.py`` on the result.

Usage (from the repo root)::

    python tools/test_selection/run_baseline.py
    python tools/test_selection/run_baseline.py -- -n 8          # extra pytest args
    python tools/test_selection/run_baseline.py --paths tests/test_notebook_integration/basics/test_basic_flow.py

Coverage slows the kernels down, so the per-test timeout is raised to 90 s for
this run. Everything else (workers, reruns) comes from pyproject's addopts.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

RC_TEMPLATE = """\
[run]
parallel = true
branch = false
sigterm = true
source = cash
data_file = {data_file}
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / ".testsel"))
    ap.add_argument("--paths", nargs="*", default=["tests/test_notebook_integration"])
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--no-select", action="store_true")
    ap.add_argument("pytest_args", nargs="*")
    args = ap.parse_args()

    out = Path(args.out).resolve()
    cov_dir = out / "cov"
    cov_dir.mkdir(parents=True, exist_ok=True)
    for stale in cov_dir.glob(".coverage*"):
        stale.unlink()

    rc = out / "coverage.rc"
    rc.write_text(RC_TEMPLATE.format(data_file=(cov_dir / ".coverage").as_posix()), encoding="utf-8")

    env = dict(os.environ)
    env["COVERAGE_PROCESS_START"] = str(rc)
    env["PYTHONPATH"] = os.pathsep.join([str(HERE), str(ROOT)] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    env["TESTSEL_RESULTS"] = str(out / "results.json")

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        *args.paths,
        "-p",
        "covctx_plugin",
        f"--timeout={args.timeout}",
        "-q",
        "-rfE",
        "-p",
        "no:cacheprovider",
        *args.pytest_args,
    ]
    commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    print("running:", " ".join(cmd), flush=True)

    t0 = time.time()
    with open(out / "pytest.log", "w", encoding="utf-8") as log:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        for line in proc.stdout:
            sys.stdout.write(line)
            log.write(line)
        code = proc.wait()
    elapsed = time.time() - t0

    combine = subprocess.run(
        [sys.executable, "-m", "coverage", "combine", f"--rcfile={rc}"], cwd=ROOT, capture_output=True, text=True
    )
    print(combine.stdout[-2000:], combine.stderr[-2000:])

    info = {
        "commit": commit,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "paths": args.paths,
        "pytest_args": args.pytest_args,
        "timeout": args.timeout,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t0)),
        "elapsed_seconds": round(elapsed, 1),
        "pytest_exit_code": code,
        "coverage_combine_ok": combine.returncode == 0,
    }
    (out / "run_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(json.dumps(info, indent=2))

    if not args.no_select:
        subprocess.run([sys.executable, str(HERE / "select_core.py"), "--out", str(out)], cwd=ROOT)
    # pytest's own exit code, so a caller that checks only ours still sees a failed run.
    return code


if __name__ == "__main__":
    raise SystemExit(main())
