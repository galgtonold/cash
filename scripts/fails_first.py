"""Prove a new test can actually fail: run it against the UNFIXED source.

    python scripts/fails_first.py tests/test_core/test_module_attr_dependencies.py

Checks out ``HEAD`` into a temporary git worktree, copies over your uncommitted
changes OUTSIDE ``src/`` (the new tests, conftest edits, test data), and runs the
given tests there with that worktree's ``src/`` first on ``PYTHONPATH``. So the
tests are yours and the source is the last commit, without the fix. Your own
tree, index and stashes are never touched. A test that passes here is not
testing your fix -- it is vacuously green.

Vacuous green has four shapes, all of which have shipped in this repo:

1. The mechanism never engages. A decorator test whose function runs faster
   than the ~0.1 s persistence floor never writes to disk, so a second process
   recomputes and the assertion holds whether or not the bug exists. This one
   silently passed two brand-new tests against unfixed source.
2. Empty input trivially satisfies the assertion. "Is this output encodable?"
   passes for an empty string, so a harness that failed to execute anything
   looks like a pass. Assert the input is non-empty first.
3. A different gate is substituted for the real one. ``mkdocs build --strict``
   validates links and nav; it never executes a python fence, so docs whose
   code raises NameError build green.
4. State is checked instead of behaviour. Asserting a policy object exists
   passes even when nothing ever calls it. Drive the behaviour on both sides
   of the boundary.

Fail-first only catches (1) and (4) directly, but it catches them cheaply and
without any judgement call, which is why it is worth automating.

Exit codes: 0 when every selected test failed without the fix; 1 when at least
one passed (or was skipped) without it, which names those tests; 2 when nothing
could be judged -- no uncommitted change under ``src/``, a collection or usage
error, no tests selected, or a git failure.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _git(*args: str, cwd: Path = REPO) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True)


def _changed_outside_src() -> list[str]:
    """Paths outside src/ that differ from HEAD: tracked edits (staged or not),
    deletions, and untracked files that are not ignored."""
    tracked = _git("diff", "--name-only", "--no-renames", "-z", "HEAD").stdout.split("\0")
    untracked = _git("ls-files", "--others", "--exclude-standard", "-z").stdout.split("\0")
    return sorted({p for p in tracked + untracked if p and not p.startswith("src/")})


def _outcomes(junit: Path) -> dict[str, str]:
    """Map each test's id to passed / failed / skipped from a JUnit report."""
    outcomes: dict[str, str] = {}
    for case in ET.parse(junit).iter("testcase"):
        name = f"{case.get('classname', '')}::{case.get('name', '')}"
        if case.find("failure") is not None or case.find("error") is not None:
            outcomes[name] = "failed"
        elif case.find("skipped") is not None:
            outcomes.setdefault(name, "skipped")
        else:
            outcomes.setdefault(name, "passed")
    return outcomes


def _in_repo(target: str) -> str:
    """Rewrite a test path given from the current directory relative to the
    repo, so it names the same test inside the temporary worktree."""
    path, sep, rest = target.partition("::")
    full = Path(path).resolve()
    if not full.exists() or not full.is_relative_to(REPO):
        return target
    return full.relative_to(REPO).as_posix() + sep + rest


def main() -> int:
    targets = [_in_repo(t) for t in sys.argv[1:]]
    if not targets:
        print(__doc__)
        return 2

    # --untracked-files=all so a fix that only ADDS a module still counts.
    dirty = _git("status", "--porcelain", "--untracked-files=all", "--", "src/").stdout.strip()
    if not dirty:
        print("NOTE: no uncommitted changes under src/ -- the last commit already")
        print("      holds the fix, so there is nothing to compare against.")
        print("      Check the fix out to a scratch branch, move the source change")
        print("      back into the working tree, and run this again.")
        return 2

    tmp = Path(tempfile.mkdtemp(prefix="fails_first-"))
    unfixed = tmp / "tree"
    try:
        added = _git("worktree", "add", "--detach", "--quiet", str(unfixed), "HEAD")
        if added.returncode != 0:
            print("ERROR: could not check out HEAD into a temporary worktree:")
            print(added.stderr.strip())
            return 2
        for rel in _changed_outside_src():
            src, dst = REPO / rel, unfixed / rel
            if src.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
            elif dst.is_file() and not src.exists():
                dst.unlink()

        junit = tmp / "junit.xml"
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(unfixed / "src"), env.get("PYTHONPATH")]))
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                *targets,
                "-n0",
                "-p",
                "no:randomly",
                "-q",
                "--tb=no",
                f"--junitxml={junit}",
            ],
            cwd=str(unfixed),
            env=env,
            capture_output=True,
            text=True,
        )
        outcomes = _outcomes(junit) if junit.is_file() else {}
    finally:
        if _git("worktree", "remove", "--force", str(unfixed)).returncode != 0:
            _git("worktree", "prune")
        shutil.rmtree(tmp, ignore_errors=True)

    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    print("Result without the fix:", (lines[-1:] or ["<no summary>"])[0])

    # pytest: 0 all passed, 1 some failed, 2 interrupted (collection error),
    # 3 internal error, 4 usage error (e.g. a node id not found), 5 no tests.
    if proc.returncode not in (0, 1) or not outcomes:
        print()
        print(f"INCONCLUSIVE: pytest exited {proc.returncode} without running the tests.")
        print("A collection error proves nothing about the assertions. If the test")
        print("imports a name the fix adds, import it inside the test instead.")
        print("\n".join(lines[-15:]))
        return 2

    not_failed = {name: kind for name, kind in outcomes.items() if kind != "failed"}
    for name, kind in outcomes.items():
        print(f"  {kind.upper():8} {name}")
    if not_failed:
        print()
        print("VACUOUS: these tests did not fail without the fix:")
        for name, kind in not_failed.items():
            print(f"  {name} ({kind})")
        print("They do not exercise the change. Common cause: the mechanism")
        print("never engages -- e.g. a cached function under the ~0.1 s")
        print("persistence floor is never written to disk at all.")
        return 1

    print()
    print(f"OK: all {len(outcomes)} selected tests failed without the fix.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
