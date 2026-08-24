"""Drift test: committed brand PNGs must match the HTML they were rendered from.

If this fails, run:

    python scripts/build_brand_assets.py

and commit the regenerated docs/_brand/*.png (and the .stamp beside each).

Why these assets in particular need a guard: the social card is set once in the
repository's Settings and never appears anywhere a maintainer looks, and the
README header is seen so constantly that it stops being read. Either could go
on advertising an old positioning -- or an old install name -- indefinitely,
because nothing you look at day to day would be wrong.

Same chain of custody, and the same reasoning, as test_badge_images_fresh: hash
the *source*, not the pixels. Headless screenshots are not byte-stable across
platforms (font hinting differs), so a pixel comparison would fail for reasons
that have nothing to do with an asset being stale.

The asset table is read from the build script rather than restated here. Two
variants share one source (the header renders light and dark from the same
HTML), so a png -> html mapping guessed from filenames is wrong by
construction -- and a second copy of the table is one more thing that can
disagree with the first.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BRAND_DIR = REPO_ROOT / "docs" / "_brand"
BADGE_SCRIPT = REPO_ROOT / "scripts" / "build_badge_images.py"
BRAND_SCRIPT = REPO_ROOT / "scripts" / "build_brand_assets.py"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _stamper():
    """The stamp function the builders actually use.

    Deliberately build_badge_images': build_brand_assets imports its stamp from
    there rather than defining a second one. Two stamp implementations that
    must agree is exactly the shape that reddened CI once already, so there is
    only ever one.
    """
    return _load(BADGE_SCRIPT, "build_badge_images")._stamp


def _assets():
    return _load(BRAND_SCRIPT, "build_brand_assets").ASSETS


def test_there_are_brand_assets():
    """Guards against the parametrize below silently passing on an empty set."""
    assert _assets(), "build_brand_assets.ASSETS is empty"
    assert sorted(BRAND_DIR.glob("*.png")), "no docs/_brand/*.png found"


@pytest.mark.parametrize(
    "src, out_stem", [(a[0], a[1]) for a in _assets()], ids=lambda v: str(v)
)
def test_png_matches_the_html_it_was_rendered_from(src: str, out_stem: str):
    html = BRAND_DIR / f"{src}.html"
    png = BRAND_DIR / f"{out_stem}.png"
    stamp = BRAND_DIR / f"{out_stem}.png.stamp"

    assert html.exists(), f"{out_stem} names a source {html.name} that does not exist"
    assert png.exists(), f"{png.name} has not been built"
    assert stamp.exists(), (
        f"{png.name} has no .stamp recording which HTML it came from. "
        "Re-run `python scripts/build_brand_assets.py`."
    )
    assert stamp.read_text(encoding="utf-8").strip() == _stamper()(html), (
        f"{png.name} was rendered from an older {html.name}. Re-run "
        "`python scripts/build_brand_assets.py` and commit the new PNG."
    )


def test_every_committed_png_is_one_the_builder_owns():
    """An orphan PNG has no source and so can never be found stale."""
    owned = {f"{a[1]}.png" for a in _assets()}
    found = {p.name for p in BRAND_DIR.glob("*.png")}
    assert found == owned, (
        f"docs/_brand PNGs do not match what the builder produces. "
        f"orphans: {sorted(found - owned)}, missing: {sorted(owned - found)}"
    )


def test_declared_sizes_match_the_committed_files():
    """The dimensions that matter are the ones in the file, not in the script.

    Reading the real PNG header catches a viewport or scale change that was
    never re-rendered -- which the stamp cannot, since the stamp only covers
    the HTML.
    """
    for _src, out_stem, width, height, scale, _scheme in _assets():
        actual = _png_size(BRAND_DIR / f"{out_stem}.png")
        expected = (width * scale, height * scale)
        assert actual == expected, (
            f"{out_stem}.png is {actual[0]}x{actual[1]}, expected "
            f"{expected[0]}x{expected[1]} (declared {width}x{height} at {scale}x)"
        )


def test_the_social_card_stays_inside_githubs_upload_limits():
    """GitHub caps the social preview at 1MB and documents 1280x640.

    A 2x card came out 2560x1280 at 822KB -- inside the stated size limit, and
    the upload still would not take, leaving the Settings preview blank with no
    error worth noticing. The card is rendered at 1x for that reason, and this
    pins it: a silent upload failure is expensive to diagnose precisely because
    nothing anywhere reports it.
    """
    png = BRAND_DIR / "social-card.png"
    size_kb = png.stat().st_size / 1024
    assert size_kb < 1024, f"social-card.png is {size_kb:.0f}KB; GitHub caps at 1MB"

    width, height = _png_size(png)
    assert (width, height) == (1280, 640), (
        f"social-card.png is {width}x{height}; GitHub documents 1280x640 for "
        f"the social preview, and deviating from it is what broke the upload."
    )


def test_the_social_card_keeps_githubs_two_to_one_ratio():
    width, height = _png_size(BRAND_DIR / "social-card.png")
    ratio = width / height
    assert abs(ratio - 2.0) < 0.01, (
        f"social-card.png ratio is {ratio:.3f}; GitHub expects 2:1 and crops "
        f"anything else."
    )
    assert width >= 1280, f"{width}px wide is below GitHub's 1280 minimum"


def _png_size(path: Path) -> tuple[int, int]:
    """Width and height from the IHDR chunk -- no image library needed."""
    header = path.read_bytes()[:24]
    assert header[:8] == b"\x89PNG\r\n\x1a\n", f"{path.name} is not a PNG"
    return (int.from_bytes(header[16:20], "big"), int.from_bytes(header[20:24], "big"))
