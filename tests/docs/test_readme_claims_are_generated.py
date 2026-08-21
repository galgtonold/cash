"""The README's two load-bearing figures must trace back to real sources.

The README shows a Cash badge and quotes a restore cost. Both were prose
before: the badge was *described* ("Cash shows a badge above each cell"), one
link away from ever being seen, and the page carried no measured number at all.

Showing them is better -- but sample output is exactly the kind of thing that
rots silently, and the badge vocabulary has churned before (the CAS-272
restored/computed -> CACHED/EXECUTED rename left 22 stale assertions behind).
So neither is hand-written:

* the badge is a PNG built from committed badge HTML by
  ``scripts/build_badge_images.py``. This test checks the README points at a
  PNG that exists; ``tests/test_docs/test_badge_images_fresh.py`` checks that
  PNG still matches its HTML, and ``test_badge_examples_fresh`` checks that
  HTML still matches the renderer.
* the restore cost is pinned to the same frozen matrix that
  ``test_benchmarks_table_matches_frozen_data`` holds ``docs/benchmarks.md``
  to, so the README and the benchmarks page cannot drift apart from each other
  or from the measurement.
"""
from __future__ import annotations

import csv
import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[2]
README = REPO / "README.md"
FROZEN = REPO / "benchmarks" / "results" / "ser_deser_matrix.frozen.csv"

# The README must use an absolute URL: PyPI renders the description standalone
# and cannot resolve repo-relative image paths.
_RAW_PREFIX = "https://raw.githubusercontent.com/galgtonold/cash/main/"


@pytest.fixture(scope="module")
def readme() -> str:
    return README.read_text(encoding="utf-8")


def test_readme_badge_image_is_committed(readme):
    """The badge picture must be an absolute URL to a PNG that exists here."""
    urls = re.findall(r"!\[[^\]]*\]\((https?://[^)]+\.png)\)", readme)
    badge = [u for u in urls if "_badges/" in u]
    assert badge, (
        "the README no longer embeds a badge image from docs/_badges/. If the "
        "badge was moved or replaced, update this test with it -- do not delete it."
    )
    for url in badge:
        assert url.startswith(_RAW_PREFIX), (
            f"{url} is not an absolute raw.githubusercontent URL. PyPI renders the "
            "README standalone and cannot resolve repo-relative image paths."
        )
        local = REPO / url[len(_RAW_PREFIX):]
        assert local.exists(), (
            f"README points at {url}, but {local.relative_to(REPO)} is not in the "
            "repo. Run `python scripts/build_badge_images.py` and commit the PNG."
        )


def test_readme_badge_image_has_alt_text(readme):
    """A picture carrying the explanation needs a description for screen readers."""
    for alt, url in re.findall(r"!\[([^\]]*)\]\((https?://[^)]+\.png)\)", readme):
        if "_badges/" not in url:
            continue
        assert len(alt) > 20, f"badge image {url} has thin alt text: {alt!r}"


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
