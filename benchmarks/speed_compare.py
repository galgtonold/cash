"""A/B: run the speed set on two versions of cash and flag the ratios that moved.

Each ref is checked out into a temporary git worktree, and the speed set from
THIS checkout runs against that worktree's ``src/`` (``PYTHONPATH``; no venv
per side, the same interpreter and packages for both). The runs alternate
A B B A A B ... so drift on the machine (other jobs, thermal) falls on both
sides alike. Per scenario and round, the change is B's ratio over A's ratio
from the two adjacent runs; the verdict uses the median over rounds.

A scenario is a REGRESSION when B's ratio is more than ``--threshold`` (20%)
above A's and every round agrees; IMPROVED for the mirror case; BROKEN when
the scenario raised on B but not on A. Exit code 1 when anything regressed or
broke, 2 on a usage or run error, else 0.

Usage:
    python benchmarks/speed_compare.py                       # main vs HEAD
    python benchmarks/speed_compare.py 51ccdcf HEAD --rounds 3
    python benchmarks/speed_compare.py main WORKTREE         # uncommitted changes
    python benchmarks/speed_compare.py --from-json a.json b.json

``WORKTREE`` means this checkout's ``src/`` as it is on disk.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEED_SET = HERE / "speed_set.py"
WORKTREE = "WORKTREE"
FAILING = ("regression", "broken")


@dataclass
class Verdict:
    name: str
    ratio_a: float
    ratio_b: float
    change: float  # median over rounds of ratio_b / ratio_a
    rounds: list[float]
    # "regression", "improved", "same", "broken" (errors on B only), "only in A", "only in B", "skipped"
    verdict: str
    note: str = ""


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, capture_output=True, text=True, check=True, encoding="utf-8"
    ).stdout.strip()


def repo_root() -> Path:
    return Path(_git(HERE, "rev-parse", "--show-toplevel"))


class Checkout:
    """``src/`` of one ref: a temporary detached worktree, or this checkout."""

    def __init__(self, repo: Path, ref: str, parent: Path, label: str):
        self.ref = ref
        self.repo = repo
        self.path: Path | None = None
        if ref == WORKTREE:
            self.sha = "worktree"
            self.src = repo / "src"
            return
        self.sha = _git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}")
        self.path = parent / label
        _git(repo, "worktree", "add", "--detach", str(self.path), self.sha)
        self.src = self.path / "src"

    def remove(self) -> None:
        if self.path is None:
            return
        try:
            _git(self.repo, "worktree", "remove", "--force", str(self.path))
        except subprocess.CalledProcessError:
            shutil.rmtree(self.path, ignore_errors=True)
            subprocess.run(["git", "worktree", "prune"], cwd=self.repo, capture_output=True, check=False)


def run_speed_set(src: Path, out: Path, repeats: int, only: list[str] | None, speed_set: Path = SPEED_SET) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(src)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    cmd = [sys.executable, str(speed_set), "--repeats", str(repeats), "--json", str(out), "-q"]
    if only:
        cmd += ["--only", *only]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"speed set failed against {src} (exit {proc.returncode}):\n{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}"
        )
    result = json.loads(out.read_text(encoding="utf-8"))
    cash_path = Path(result["meta"]["cash_path"]).resolve()
    if Path(src).resolve() not in cash_path.parents:
        raise RuntimeError(
            f"the speed set imported cash from {cash_path}, not from {src}; is cash on PYTHONPATH overridden?"
        )
    return result


def _ratios(result: dict) -> dict[str, dict]:
    return {r["name"]: r for r in result["rows"]}


def compare(runs_a: list[dict], runs_b: list[dict], threshold: float) -> list[Verdict]:
    """Pair run i of A with run i of B (they ran back to back) and judge each row."""
    a_rows = [_ratios(r) for r in runs_a]
    b_rows = [_ratios(r) for r in runs_b]
    names = list(dict.fromkeys([n for rows in a_rows + b_rows for n in rows]))
    out = []
    for name in names:
        in_a = all(name in rows for rows in a_rows)
        in_b = all(name in rows for rows in b_rows)
        if not in_a or not in_b:
            out.append(
                Verdict(name, float("nan"), float("nan"), float("nan"), [], "only in B" if in_b else "only in A")
            )
            continue
        a_skip = [rows[name]["skipped"] for rows in a_rows if rows[name].get("skipped")]
        b_skip = [rows[name]["skipped"] for rows in b_rows if rows[name].get("skipped")]
        if b_skip and not a_skip and b_skip[0].startswith("error"):
            # Ran on A, fails on B: as bad as any slowdown.
            out.append(Verdict(name, float("nan"), float("nan"), float("nan"), [], "broken", b_skip[0]))
            continue
        if a_skip or b_skip:
            out.append(Verdict(name, float("nan"), float("nan"), float("nan"), [], "skipped", (a_skip + b_skip)[0]))
            continue
        ra = [rows[name]["ratio"] for rows in a_rows]
        rb = [rows[name]["ratio"] for rows in b_rows]
        per_round = [b / a for a, b in zip(ra, rb) if a > 0]
        change = statistics.median(per_round) if per_round else float("nan")
        verdict = "same"
        if per_round and change > 1 + threshold and all(c > 1 for c in per_round):
            verdict = "regression"
        elif per_round and change < 1 / (1 + threshold) and all(c < 1 for c in per_round):
            verdict = "improved"
        notes = sorted({n for rows in a_rows + b_rows for n in rows[name].get("notes", [])})
        out.append(
            Verdict(name, statistics.median(ra), statistics.median(rb), change, per_round, verdict, "; ".join(notes))
        )
    return out


def format_verdicts(verdicts: list[Verdict], label_a: str, label_b: str, threshold: float) -> str:
    head = f"{'scenario':<34} {'A ratio':>9} {'B ratio':>9} {'B/A':>7}  {'verdict':<11} rounds"
    lines = [f"A = {label_a}   B = {label_b}   threshold {threshold:.0%}", head, "-" * len(head)]
    for v in verdicts:
        if v.verdict in ("only in A", "only in B", "skipped", "broken"):
            lines.append(f"{v.name:<34} {'-':>9} {'-':>9} {'-':>7}  {v.verdict:<11} {v.note}")
            continue
        flag = {"regression": "REGRESSION", "improved": "improved"}.get(v.verdict, "")
        rounds = " ".join(f"{c:.2f}" for c in v.rounds)
        note = f"  {v.note}" if v.note else ""
        lines.append(f"{v.name:<34} {v.ratio_a:>8.2f}x {v.ratio_b:>8.2f}x {v.change:>7.2f}  {flag:<11} {rounds}{note}")
    reg = [v.name for v in verdicts if v.verdict in FAILING]
    imp = [v.name for v in verdicts if v.verdict == "improved"]
    lines.append("")
    lines.append(f"{len(reg)} regressed: {', '.join(reg) or '-'}")
    lines.append(f"{len(imp)} improved: {', '.join(imp) or '-'}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("a", nargs="?", default="main", help="baseline ref (default main)")
    parser.add_argument(
        "b", nargs="?", default="HEAD", help=f"candidate ref (default HEAD; {WORKTREE} = this checkout)"
    )
    parser.add_argument("--rounds", type=int, default=2, help="A/B pairs, order alternating (default 2: A B B A)")
    parser.add_argument("--repeats", type=int, default=5, help="samples per side per speed-set run (default 5)")
    parser.add_argument("--threshold", type=float, default=0.20, help="flag a ratio change above this (default 0.20)")
    parser.add_argument("--only", nargs="+", metavar="PART", help="only scenarios whose name contains PART")
    parser.add_argument("--json", type=Path, help="write every run and the verdicts here")
    parser.add_argument(
        "--from-json",
        nargs=2,
        type=Path,
        metavar=("A_JSON", "B_JSON"),
        help="compare two saved speed_set.py --json results instead of running",
    )
    parser.add_argument("--speed-set", type=Path, default=SPEED_SET, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    if args.from_json:
        a_run, b_run = (json.loads(path.read_text(encoding="utf-8")) for path in args.from_json)
        verdicts = compare([a_run], [b_run], args.threshold)
        print(format_verdicts(verdicts, str(args.from_json[0]), str(args.from_json[1]), args.threshold))
        return 1 if any(v.verdict in FAILING for v in verdicts) else 0

    try:
        repo = repo_root()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        print(f"not in a git checkout: {exc}", file=sys.stderr)
        return 2
    parent = Path(tempfile.mkdtemp(prefix="cash-speed-ab-"))
    checkouts: list[Checkout] = []
    try:
        try:
            checkouts = [Checkout(repo, args.a, parent, "a")]
            checkouts.append(Checkout(repo, args.b, parent, "b"))
        except subprocess.CalledProcessError as exc:
            print(f"cannot check out {exc.cmd[-1]}: {exc.stderr.strip()}", file=sys.stderr)
            return 2
        a, b = checkouts
        label_a, label_b = f"{args.a} ({a.sha[:9]})", f"{args.b} ({b.sha[:9]})"
        runs: dict[str, list[dict]] = {"a": [], "b": []}
        for r in range(args.rounds):
            order = (("a", a), ("b", b)) if r % 2 == 0 else (("b", b), ("a", a))
            for key, co in order:
                print(f"round {r + 1}/{args.rounds}: {key.upper()} = {co.ref}", file=sys.stderr, flush=True)
                out = parent / f"{key}-{r}.json"
                try:
                    runs[key].append(run_speed_set(co.src, out, args.repeats, args.only, args.speed_set))
                except RuntimeError as exc:
                    print(str(exc), file=sys.stderr)
                    return 2
        verdicts = compare(runs["a"], runs["b"], args.threshold)
        print(format_verdicts(verdicts, label_a, label_b, args.threshold))
        if args.json:
            args.json.parent.mkdir(parents=True, exist_ok=True)
            args.json.write_text(
                json.dumps(
                    {
                        "a": label_a,
                        "b": label_b,
                        "threshold": args.threshold,
                        "runs_a": runs["a"],
                        "runs_b": runs["b"],
                        "verdicts": [v.__dict__ for v in verdicts],
                    },
                    indent=1,
                ),
                encoding="utf-8",
            )
        return 1 if any(v.verdict in FAILING for v in verdicts) else 0
    finally:
        for co in checkouts:
            co.remove()
        shutil.rmtree(parent, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
