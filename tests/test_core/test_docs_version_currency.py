"""User-facing docs must not name a version that isn't the current one.

`README.md` told every GitHub visitor "`0.2.0` is the current release" for the
whole life of 0.3.0 on PyPI. Nothing caught it: the claims system pins prose to
code *symbols*, and a version string is not a symbol, so a hard-coded number in
prose drifts silently at every release. For a tool whose entire pitch is "trust
me about staleness", the landing page being stale is a self-inflicted wound.

Two rules, and the first is the important one:

* Prose should not restate the version at all. The README already carries a
  live PyPI version badge, so a number written into the sentence is duplicating
  something that updates itself. Removing it deletes the drift class rather
  than resetting the clock.
* Where a version genuinely has to appear -- sample CLI output, which is only
  useful if it looks like what the reader will actually see -- it must match
  ``cash.__version__``, the single source of truth that `pyproject.toml` reads
  at build time.

So this test does not ask "is the number right?"; it asks "is there a number
here at all, and if so does it match?". A release that bumps `__version__`
without touching these files fails here instead of shipping.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

import cash

REPO_ROOT = Path(__file__).resolve().parents[2]

# Files a prospective user reads before they trust the project. Scoped
# deliberately: CHANGELOG and versioning.md are FULL of version numbers that
# are supposed to be historical or illustrative, and sweeping those in would
# make this test noise.
USER_FACING = (
    "README.md",
    "docs/faq.md",
    "docs/cli.md",
    "docs/index.md",
    "docs/getting-started/quickstart.md",
    "docs/getting-started/installation.md",
)

# Optional "v" prefix, because `# Cash v0.2.0` slipped past an earlier sweep
# whose pattern started with a word boundary.
_VERSION = re.compile(r"v?(\d+\.\d+\.\d+)")

# Numbers that are legitimately not cash's own version. Each needs a reason --
# an unexplained entry here is how a real drift gets waved through.
ALLOWED = {
    "3.10.0",   # Python version floors
    "3.11.0",
    "3.12.0",
    "3.13.0",
    "3.14.0",
}


@pytest.mark.parametrize("relpath", USER_FACING)
def test_no_doc_names_a_version_other_than_the_current_one(relpath: str) -> None:
    path = REPO_ROOT / relpath
    if not path.exists():  # pragma: no cover - keeps the list resilient to renames
        pytest.skip(f"{relpath} does not exist")

    text = path.read_text(encoding="utf-8")
    stale = []
    for line_no, line in enumerate(text.splitlines(), 1):
        for match in _VERSION.finditer(line):
            found = match.group(1)
            if found == cash.__version__ or found in ALLOWED:
                continue
            stale.append(f"  {relpath}:{line_no}  {found!r}  in: {line.strip()[:110]}")

    assert not stale, (
        f"cash.__version__ is {cash.__version__!r}, but these user-facing lines "
        "name a different version:\n" + "\n".join(stale) + "\n\n"
        "Either update the line, or -- better -- delete the number. Prose that "
        "restates the version duplicates the PyPI badge and drifts at every "
        "release. Only sample output needs a literal version."
    )


def test_the_readme_does_not_restate_the_version_in_prose() -> None:
    """The stronger rule, and the one that actually removes the drift class.

    Passing the test above by bumping the number each release is a treadmill
    someone eventually steps off. The README carries a live PyPI badge; the
    sentence does not need to agree with it, it needs to not compete with it.
    """
    text = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    offenders = [
        line.strip()
        for line in text.splitlines()
        if _VERSION.search(line) and "img.shields.io" not in line
    ]
    assert not offenders, (
        "README prose names a version literal; let the PyPI badge be the single "
        "place a reader learns the current version:\n  " + "\n  ".join(offenders)
    )


def test_the_version_source_of_truth_is_importable() -> None:
    """The control. Without it, a broken import would make both tests above
    vacuous -- every comparison would trivially pass or the file would skip."""
    assert re.fullmatch(r"\d+\.\d+\.\d+", cash.__version__), cash.__version__
