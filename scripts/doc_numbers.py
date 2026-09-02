#!/usr/bin/env python3
"""Derive the numbers the docs quote, and fail the build when they drift.

Every hand-maintained figure in the documentation is a fact about the
repository that nobody re-derives: the version, how many tests there are, how
many claim anchors, how many platform combinations CI runs. They were all
wrong at once when this script was written -- the README said ~8,500 tests
while ``testing.md`` said ~8,750 and, three sections later, ~8,800; the claim
count said 164 against an actual 247; ``versioning.md`` pinned ``~=0.2.0`` six
minor versions after the fact. None of it was caught by anything, because
prose is not executed and a stale number still renders.

**Why a generator rather than a template placeholder.** ``mkdocs`` could
substitute ``{{ tests_total }}`` at build time, but the README is rendered by
PyPI and GitHub, which run no build at all -- a placeholder there would ship
literally. And the expensive facts (test counts) need a pytest collection pass,
which is a CI-time thing, not something to run on every docs preview. So the
values are **committed** in the markdown, exactly as a reader sees them, and
this script is what proves they are still true.

## The marker

::

    Roughly <!-- docnum:tests_total -->~8,750<!-- /docnum --> tests.

HTML comments are invisible in all three renderers (GitHub, PyPI's
``readme_renderer``, and python-markdown), so the marked text reads as ordinary
prose everywhere while staying machine-addressable.

## Rounding is deliberate

A number that moves every time somebody adds a test would make this gate a
nuisance and the docs a diff-noise generator, so counts are rounded to a step
and rendered with a ``~``. The error is bounded at half a step and the value
only changes when the underlying fact meaningfully has.

## Usage

::

    python scripts/doc_numbers.py --list      # show every fact and its value
    python scripts/doc_numbers.py --check     # exit 1 on drift (CI)
    python scripts/doc_numbers.py --update    # rewrite the markdown in place

``--fast`` skips the facts that need a pytest collection pass, which is what
the docs test suite runs so version/claim/platform drift is caught on every
ordinary test run without paying for three collections.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

#: Files scanned for markers. Everything published, plus the README, which is
#: the one page rendered by PyPI and GitHub rather than by mkdocs.
SCANNED = ["README.md", *sorted(str(p.relative_to(REPO).as_posix())
                                for p in (REPO / "docs").rglob("*.md"))]


def _scanned(root: Path) -> list[str]:
    """The marked files, relative to ``root``.

    Recomputed rather than reusing ``SCANNED`` so a caller can point this at a
    COPY of the docs tree. The drift test needs to corrupt a marked file to
    prove the checker fails on drift, and corrupting the real one races every
    other test that reads it -- under xdist those run in separate processes,
    so a `finally` that restores the file is not protection.
    """
    if root == REPO:
        return SCANNED
    return ["README.md", *sorted(str(q.relative_to(root).as_posix())
                                 for q in (root / "docs").rglob("*.md"))]

MARKER = re.compile(
    r"<!--\s*docnum:(?P<name>[a-z0-9_]+)\s*-->(?P<value>.*?)<!--\s*/docnum\s*-->",
    re.DOTALL,
)

#: The canonical three-way suite split, kept identical to the commands
#: ``docs/how-it-works/testing.md`` tells a reader to run. If those diverge the
#: page is describing a suite nobody can reproduce.
SUITES = {
    "tests_unit": ["tests/", "--ignore=tests/test_notebook_integration",
                   "--ignore=tests/test_wheel_gate", "--ignore=tests/docs"],
    "tests_integration": ["tests/test_notebook_integration"],
    "tests_docs": ["tests/docs"],
}

_COLLECTED = re.compile(r"(\d+)\s+tests?\s+collected")


# --------------------------------------------------------------------------
# Deriving the facts
# --------------------------------------------------------------------------

def _read_version() -> str:
    """The single source of truth, per pyproject's own comment."""
    text = (REPO / "src" / "cash" / "__init__.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__\s*=\s*"([^"]+)"', text, re.MULTILINE)
    if match is None:
        raise SystemExit("could not find __version__ in src/cash/__init__.py")
    return match.group(1)


def _collect(args: list[str]) -> int:
    """Count tests without running them."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", *args, "--collect-only", "-q",
         "-p", "no:randomly"],
        cwd=REPO, capture_output=True, text=True,
    )
    hits = _COLLECTED.findall(proc.stdout)
    if not hits:
        raise SystemExit(
            f"could not parse a test count from `pytest {' '.join(args)}`:\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )
    return int(hits[-1])


def _count_test_files() -> int:
    return sum(1 for p in (REPO / "tests").rglob("test_*.py"))


def _count_claims() -> int:
    """Claim anchors across the documentation.

    Counts the ``<!-- claim:`` comments, not the symbols inside them -- one
    anchor can pin several functions, and the page's sentence is "N claims".
    """
    total = 0
    for rel in SCANNED:
        total += (REPO / rel).read_text(encoding="utf-8").count("<!-- claim:")
    return total


def _count_platforms() -> int:
    """Size of the CI test matrix: operating systems x Python versions."""
    text = (REPO / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    oses = re.search(r"^\s*os:\s*\[([^\]]+)\]", text, re.MULTILINE)
    pys = re.search(r"^\s*python-version:\s*\[([^\]]+)\]", text, re.MULTILINE)
    if oses is None or pys is None:
        raise SystemExit("could not parse the CI matrix from ci.yml")
    n_os = len([x for x in oses.group(1).split(",") if x.strip()])
    n_py = len([x for x in pys.group(1).split(",") if x.strip()])
    return n_os * n_py


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def _approx(step: int):
    """Render a count rounded to *step*, prefixed with ``~``.

    Rounding is what keeps this gate from firing on every added test. Half a
    step is the worst the printed figure can be wrong by.
    """
    def render(value: int) -> str:
        return f"~{round(value / step) * step:,}"
    return render


def _exact(value: int) -> str:
    return f"{value:,}"


def _pin(version: str) -> str:
    """The compatible-release pin a reader should copy for this version."""
    major, minor, *_ = version.split(".")
    return f"~={major}.{minor}.0"


def _next_minor(version: str) -> str:
    """The version a compatible-release pin deliberately will NOT take."""
    major, minor, *_ = version.split(".")
    return f"{major}.{int(minor) + 1}"


class Fact:
    def __init__(self, name, compute, render=str, expensive=False):
        self.name = name
        self._compute = compute
        self._render = render
        self.expensive = expensive

    def value(self) -> str:
        return self._render(self._compute())


def _build_facts() -> dict[str, Fact]:
    version = _read_version()
    facts = [
        Fact("version", lambda: version),
        Fact("version_pin", lambda: _pin(version)),
        Fact("version_next_minor", lambda: _next_minor(version)),
        Fact("platforms", _count_platforms, _exact),
        Fact("claims", _count_claims, _approx(10)),
        Fact("test_files", _count_test_files, _approx(10)),
    ]
    for name, args in SUITES.items():
        facts.append(Fact(name, lambda a=args: _collect(a), _approx(10),
                          expensive=True))
    facts.append(Fact(
        "tests_total",
        lambda: sum(_collect(a) for a in SUITES.values()),
        _approx(50), expensive=True,
    ))
    return {f.name: f for f in facts}


# --------------------------------------------------------------------------
# Applying
# --------------------------------------------------------------------------

def _resolve(facts, fast: bool) -> dict[str, str]:
    resolved = {}
    for name, fact in facts.items():
        if fast and fact.expensive:
            continue
        resolved[name] = fact.value()
    return resolved


def _walk(resolved: dict[str, str], root: Path = REPO):
    """Yield ``(path, name, current, wanted)`` for every marker in the docs."""
    for rel in _scanned(root):
        path = root / rel
        text = path.read_text(encoding="utf-8")
        for match in MARKER.finditer(text):
            name = match.group("name")
            if name not in resolved:
                continue
            yield rel, name, match.group("value"), resolved[name]


def _rewrite(resolved: dict[str, str], root: Path = REPO) -> list[str]:
    changed = []
    for rel in _scanned(root):
        path = root / rel
        text = path.read_text(encoding="utf-8")

        def sub(match):
            name = match.group("name")
            if name not in resolved:
                return match.group(0)
            return (f"<!-- docnum:{name} -->{resolved[name]}"
                    f"<!-- /docnum -->")

        new = MARKER.sub(sub, text)
        if new != text:
            path.write_text(new, encoding="utf-8")
            changed.append(rel)
    return changed


def _unknown_markers(resolved_names, root: Path = REPO) -> list[tuple[str, str]]:
    """Markers naming a fact this script does not know how to derive.

    A typo in a marker name would otherwise be invisible: the marker simply
    never updates, which is the exact failure this script exists to remove.
    """
    known = set(resolved_names)
    bad = []
    for rel in _scanned(root):
        text = (root / rel).read_text(encoding="utf-8")
        for match in MARKER.finditer(text):
            if match.group("name") not in known:
                bad.append((rel, match.group("name")))
    return bad


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true",
                      help="exit 1 if any marker is stale")
    mode.add_argument("--update", action="store_true",
                      help="rewrite markers in place")
    mode.add_argument("--list", action="store_true",
                      help="print every derived fact")
    parser.add_argument("--fast", action="store_true",
                        help="skip facts needing a pytest collection pass")
    parser.add_argument("--docs-root", type=Path, default=REPO, metavar="DIR",
                        help="read and write markers under DIR instead of the "
                             "repository (facts are still derived from the "
                             "repository; used by the drift test)")
    args = parser.parse_args()
    root = args.docs_root.resolve()

    facts = _build_facts()
    resolved = _resolve(facts, args.fast)

    if args.list:
        width = max(len(n) for n in facts)
        for name, fact in facts.items():
            value = resolved.get(name, "(skipped: --fast)")
            tag = "  [collect]" if fact.expensive else ""
            print(f"  {name:<{width}}  {value}{tag}")
        return 0

    # A misspelled marker name never updates and never fails -- catch it in
    # both modes, against the full fact list rather than the resolved subset,
    # so --fast does not report every expensive marker as a typo.
    unknown = _unknown_markers(facts, root)
    if unknown:
        print("Unknown docnum markers (no such fact):")
        for rel, name in unknown:
            print(f"  {rel}: docnum:{name}")
        return 1

    if args.update:
        changed = _rewrite(resolved, root)
        if changed:
            print("Updated:")
            for rel in changed:
                print(f"  {rel}")
        else:
            print("Already up to date.")
        return 0

    stale = [(rel, name, cur, want)
             for rel, name, cur, want in _walk(resolved, root) if cur != want]
    if stale:
        print("Stale numbers in the docs "
              "(run `python scripts/doc_numbers.py --update`):\n")
        for rel, name, cur, want in stale:
            print(f"  {rel}")
            print(f"    docnum:{name}: {cur!r} -> {want!r}")
        return 1

    counted = len(list(_walk(resolved, root)))
    scope = " (fast: skipped test counts)" if args.fast else ""
    print(f"All {counted} derived numbers are current{scope}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
