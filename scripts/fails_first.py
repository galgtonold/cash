"""Prove a new test can actually fail: run it against the UNFIXED source.

    python scripts/fails_first.py tests/test_core/test_module_attr_dependencies.py

Stashes uncommitted changes under ``src/``, runs the given tests, restores, and
reports which ones failed without the fix. A test that passes here is not
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

Exit code is 0 when EVERY selected test failed without the fix.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=str(REPO), capture_output=True, text=True)


def main() -> int:
    targets = sys.argv[1:]
    if not targets:
        print(__doc__)
        return 2

    dirty = _git("status", "--porcelain", "--", "src/").stdout.strip()
    if not dirty:
        print("NOTE: no uncommitted changes under src/ -- nothing to stash.")
        print("      If your fix is already committed, check it out to a scratch")
        print("      branch and revert the source change before running this.")
        return 2

    print("Stashing src/ ...")
    if _git("stash", "push", "--quiet", "--", "src/").returncode != 0:
        print("ERROR: could not stash src/; aborting without running anything.")
        return 2
    restored = False
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", *targets,
             "-n0", "-p", "no:randomly", "-q", "--tb=no"],
            cwd=str(REPO), capture_output=True, text=True,
        )
        out = proc.stdout
    finally:
        # Restore before reporting: never leave the tree stashed on any path,
        # including one where pytest itself raised. No `return` here -- that
        # would swallow the exception that brought us in.
        restored = _git("stash", "pop", "--quiet").returncode == 0
        print("Restored src/." if restored
              else "!! FAILED TO RESTORE src/ -- run `git stash pop` yourself !!")

    if not restored:
        return 2

    tail = [ln for ln in out.splitlines() if ln.strip()][-1:] or ["<no summary>"]
    print("Result without the fix:", tail[0])

    if proc.returncode == 0:
        print()
        print("VACUOUS: every selected test PASSED without the fix.")
        print("They do not exercise the change. Common cause: the mechanism")
        print("never engages -- e.g. a cached function under the ~0.1 s")
        print("persistence floor is never written to disk at all.")
        return 1

    print()
    print("OK: at least one test failed without the fix, so it is a real guard.")
    print("Check the names above match the tests you just wrote -- a failure in")
    print("an unrelated test does not vouch for yours.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
