"""``python -m tools.plain_checker SHAPES.py [...]``: run shapes against
plain Python and print what differs. Exit status 1 when anything does."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import runpy
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from tools.plain_checker import CONFIGS, Shape, check


def load(paths: list[str]) -> list[Shape]:
    shapes: list[Shape] = []
    for path in paths:
        shapes.extend(runpy.run_path(path)["SHAPES"])
    return shapes


def _one(args: tuple[Shape, str, list[str]]):
    shape, work, configs = args
    return check(shape, Path(work) / shape.name, configs)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m tools.plain_checker", description=__doc__)
    parser.add_argument("files", nargs="*", help="shape files (default: the starter shapes)")
    parser.add_argument("-k", dest="match", help="only shapes whose name contains this")
    parser.add_argument("--configs", default=",".join(CONFIGS), help=f"comma-separated, from {', '.join(CONFIGS)}")
    parser.add_argument("-j", "--jobs", type=int, default=1, help="shapes run at once (each starts its own kernels)")
    parser.add_argument("--work", help="folder for the runs (default: a temporary folder)")
    parser.add_argument("--statuses", action="store_true", help="print cash's statuses for passing shapes too")
    parser.add_argument("--json", dest="json_out", help="also write the reports to this file")
    parser.add_argument("--list", action="store_true", help="list the shapes and exit")
    options = parser.parse_args(argv)

    files = options.files or [str(Path(__file__).parent / "shapes" / "starter.py")]
    shapes = [s for s in load(files) if not options.match or options.match in s.name]
    if options.list:
        for shape in shapes:
            print(f"{shape.kind:9} {shape.name:40} {shape.source}")
        return 0
    configs = [c.strip() for c in options.configs.split(",") if c.strip()]
    unknown = [c for c in configs if c not in CONFIGS]
    if unknown:
        parser.error(f"unknown config(s): {', '.join(unknown)}")

    work = options.work or tempfile.mkdtemp(prefix="plain-checker-")
    jobs = [(shape, work, configs) for shape in shapes]
    reports = []
    if options.jobs > 1:
        with ProcessPoolExecutor(options.jobs, mp_context=multiprocessing.get_context("spawn")) as pool:
            for report in pool.map(_one, jobs):
                print(report.text(options.statuses), flush=True)
                reports.append(report)
    else:
        for job in jobs:
            report = _one(job)
            print(report.text(options.statuses), flush=True)
            reports.append(report)
    bad = [r for r in reports if not r.ok]
    print(f"\n{len(reports) - len(bad)} of {len(reports)} shapes match plain Python ({work})")
    if options.json_out:
        Path(options.json_out).write_text(
            json.dumps([r.__dict__ for r in reports], indent=1, default=str), encoding="utf-8"
        )
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
