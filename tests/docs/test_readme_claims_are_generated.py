"""The README's two load-bearing figures must trace back to real sources.

The README shows a Cash badge and quotes a restore cost. Both were prose
before: the badge was *described* ("Cash shows a badge above each cell") and
the page carried no measured number at all. Showing them is better -- but a
pasted sample is exactly the kind of thing that rots silently, and the badge
vocabulary has churned before (the CAS-272 restored/computed -> CACHED/EXECUTED
rename left 22 stale assertions behind).

A screenshot could not be checked this way, which is most of why the README
uses the text renderer instead: the badge is rendered from the same fixture
the docs use, so this test can regenerate it and compare byte for byte.

The restore cost is pinned to the same frozen matrix that
`test_benchmarks_table_matches_frozen_data` holds `docs/benchmarks.md` to, so
the README and the benchmarks page cannot drift apart from each other or from
the measurement.
"""
from __future__ import annotations

import csv
import importlib.util
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
README = REPO / "README.md"
FROZEN = REPO / "benchmarks" / "results" / "ser_deser_matrix.frozen.csv"
BUILDER = REPO / "scripts" / "build_badge_examples.py"

# The README's badge is this fixture, rendered as text.
_FIXTURE = "status_mixed"


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


def test_readme_badge_matches_the_renderer(readme):
    """The ```text badge block is generated output, not a hand-typed sample."""
    spec = importlib.util.spec_from_file_location("_bbe", BUILDER)
    bbe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bbe)
    from cash.notebook.badge_renderer.renderers.text import render_text

    bbe._reset_uuid_counter()
    expected = render_text(bbe.build_interactive_badge(bbe.FIXTURES[_FIXTURE])).strip()

    m = re.search(r"```text\n(\[Cash\].*?)```", readme, re.S)
    assert m, (
        "the README no longer contains a ```text block starting '[Cash]'. If the "
        "badge was moved or reformatted, update this test with it -- do not delete it."
    )
    assert m.group(1).strip() == expected, (
        "the README badge no longer matches what the text renderer produces.\n"
        f"README:\n{m.group(1).strip()}\n\nrenderer ({_FIXTURE}):\n{expected}\n\n"
        "Regenerate it rather than editing the README by hand."
    )


def test_readme_restore_cost_matches_the_frozen_matrix(readme):
    """'a 100 MB DataFrame comes back from disk in N ms' must be the measured N."""
    m = re.search(
        r"100 MB DataFrame comes back from disk in \*\*([\d.]+) ms\*\*", readme
    )
    assert m, (
        "the README no longer quotes a 100 MB DataFrame restore cost in the "
        "expected shape. If the sentence was reworded, update this test with it."
    )
    quoted = float(m.group(1))

    assert FROZEN.exists(), f"{FROZEN} is missing; it is committed on purpose."
    measured = None
    with FROZEN.open() as fh:
        for r in csv.DictReader(fh):
            if r["error"]:
                continue
            if (r["family"], r["backend_kind"], int(r["target_bytes"])) == (
                "dataframe_numeric", "disk", 100_000_000
            ):
                measured = float(r["deserialize_seconds"]) * 1000
    assert measured is not None, "no frozen row for a 100 MB DataFrame on disk"

    # Same tolerance the benchmarks-page test uses: the prose rounds.
    assert abs(measured - quoted) <= max(1.0, quoted * 0.05), (
        f"README says {quoted} ms, the frozen matrix measured {measured:.1f} ms"
    )
