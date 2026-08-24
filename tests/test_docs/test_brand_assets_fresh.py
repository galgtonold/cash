"""Drift test: the committed social card must match the HTML it was rendered from.

If this fails, run:

    python scripts/build_social_card.py

and commit the regenerated docs/_brand/*.png (and the .stamp beside it).

Why this asset in particular needs a guard: the social card is what renders when
the repository is linked from Show HN, Reddit, Discourse or Slack. It is set
once in the repo's Settings and never appears anywhere a maintainer looks, so a
card still advertising an old positioning — or an old install name — would go
unnoticed indefinitely. Nothing you see day to day would be wrong.

Same chain of custody, and the same reasoning, as test_badge_images_fresh: hash
the *source*, not the pixels. Headless screenshots are not byte-stable across
platforms (font hinting differs), so a pixel comparison would fail for reasons
that have nothing to do with the card being stale.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BRAND_DIR = REPO_ROOT / "docs" / "_brand"
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_badge_images.py"


def _stamper():
    """The generator's stamp function, so the halves cannot drift.

    Note this is *build_badge_images*, not build_social_card: the card builder
    imports its stamp from there rather than defining a second one. Two stamp
    implementations that must agree is exactly the shape that broke CI once
    already, so there is deliberately only ever one.
    """
    spec = importlib.util.spec_from_file_location("build_badge_images", BUILD_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._stamp


def _pngs():
    return sorted(BRAND_DIR.glob("*.png"))


def test_there_is_a_social_card():
    """Guards against the glob below silently passing on an empty directory."""
    assert _pngs(), (
        "no docs/_brand/*.png found. The repository's social preview is built "
        "from there; if it was removed on purpose, delete this test and "
        "scripts/build_social_card.py with it."
    )


@pytest.mark.parametrize("png", _pngs(), ids=lambda p: p.name)
def test_png_matches_the_html_it_was_rendered_from(png: Path):
    html = png.with_suffix(".html")
    stamp = Path(str(png) + ".stamp")

    assert html.exists(), f"{png.name} has no source {html.name}"
    assert stamp.exists(), (
        f"{png.name} has no .stamp recording which HTML it came from. "
        "Re-run `python scripts/build_social_card.py`."
    )
    assert stamp.read_text(encoding="utf-8").strip() == _stamper()(html), (
        f"{png.name} was rendered from an older {html.name}. Re-run "
        "`python scripts/build_social_card.py` and commit the new PNG."
    )


def test_the_card_is_the_size_github_expects():
    """GitHub renders social previews at 1280x640; anything else gets cropped.

    Checked here rather than trusted from the build script, because the value
    that matters is the one in the committed file.
    """
    png = BRAND_DIR / "social-card.png"
    width, height = _png_size(png)
    ratio = width / height
    assert abs(ratio - 2.0) < 0.01, (
        f"social-card.png is {width}x{height} (ratio {ratio:.3f}); GitHub "
        f"expects 2:1 and will crop anything else."
    )
    assert width >= 1280, f"{width}px wide is below GitHub's 1280 minimum"


def _png_size(path: Path) -> tuple[int, int]:
    """Width and height from the IHDR chunk — no image library needed."""
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} is not a PNG"
    return (int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big"))
