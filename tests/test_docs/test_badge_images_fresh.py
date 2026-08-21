"""Drift test: committed badge PNGs must match the HTML they were rendered from.

If this test fails, run:

    python scripts/build_badge_images.py

and commit the regenerated docs/_badges/*.png (and the .stamp beside it).

Why a stamp file and not a pixel comparison: re-rendering here would need
playwright on every contributor's machine and in CI, and headless screenshots
are not byte-stable across platforms anyway (font hinting differs). Hashing the
*source* HTML is deterministic everywhere and catches the thing that actually
goes wrong -- badge rendering changes and the picture in the README silently
keeps showing the old design.

Chain of custody: test_badge_examples_fresh holds the HTML to the renderer,
this holds the PNG to the HTML. Together they mean a change to badge output
cannot land while a stale picture is still committed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BADGES_DIR = REPO_ROOT / "docs" / "_badges"


def _pngs():
    return sorted(BADGES_DIR.glob("*.png"))


def test_there_is_at_least_one_badge_image():
    """Guards against the glob below silently passing on an empty set."""
    assert _pngs(), (
        "no docs/_badges/*.png found. README.md shows one; if it was removed on "
        "purpose, delete this test and scripts/build_badge_images.py with it."
    )


@pytest.mark.parametrize("png", _pngs(), ids=lambda p: p.name)
def test_png_matches_the_html_it_was_rendered_from(png: Path):
    html = png.with_suffix(".html")
    stamp = Path(str(png) + ".stamp")

    assert html.exists(), f"{png.name} has no source {html.name}"
    assert stamp.exists(), (
        f"{png.name} has no .stamp recording which HTML it came from. "
        "Re-run `python scripts/build_badge_images.py`."
    )

    expected = hashlib.sha256(html.read_bytes()).hexdigest()[:16]
    assert stamp.read_text(encoding="utf-8").strip() == expected, (
        f"{png.name} was rendered from an older {html.name}. The badge HTML has "
        "changed since -- re-run `python scripts/build_badge_images.py` and "
        "commit the new PNG."
    )
