"""Drift test: committed badge PNGs must match the HTML they were rendered from.

If this test fails, run:

    python scripts/build_badge_images.py

and commit the regenerated docs/_badges/*.png (and the .stamp beside it).

Why a stamp file and not a pixel comparison: re-rendering here would need
playwright on every contributor's machine and in CI, and headless screenshots
are not byte-stable across platforms anyway (font hinting differs). Hashing the
*source* HTML is deterministic everywhere -- once newlines are normalized, which
they must be: the HTML is CRLF in a Windows working tree and LF in every other,
so the raw bytes encode the checkout, not the badge. That hash catches the thing
that actually goes wrong -- badge rendering changes and the picture in the README
silently keeps showing the old design.

Chain of custody: test_badge_examples_fresh holds the HTML to the renderer,
this holds the PNG to the HTML. Together they mean a change to badge output
cannot land while a stale picture is still committed.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BADGES_DIR = REPO_ROOT / "docs" / "_badges"
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_badge_images.py"


def _pngs():
    return sorted(BADGES_DIR.glob("*.png"))


def _stamper():
    """The generator's own stamp function, so the two halves cannot drift.

    Importing is safe without playwright: build_badge_images.py imports it
    inside render(), not at module scope.
    """
    spec = importlib.util.spec_from_file_location("build_badge_images", BUILD_SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._stamp


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

    expected = _stamper()(html)
    assert stamp.read_text(encoding="utf-8").strip() == expected, (
        f"{png.name} was rendered from an older {html.name}. The badge HTML has "
        "changed since -- re-run `python scripts/build_badge_images.py` and "
        "commit the new PNG."
    )


def test_stamp_does_not_depend_on_the_checkout_line_endings(tmp_path):
    """A CRLF working tree must stamp identically to an LF one.

    The badge HTML reaches disk with CRLF on Windows (core.autocrlf=true on
    checkout, and the generator's write_text() in text mode) and LF everywhere
    else. While the stamp hashed raw bytes it was therefore a function of the
    platform that last ran the generator: the committed stamp was written on
    Windows, so it matched on the 2 Windows CI jobs and failed on all 10
    Linux/macOS ones -- reporting the PNG as stale when it was current.
    """
    stamp = _stamper()
    body = b"<div class='c3-wrap'>\n  <span>badge</span>\n</div>\n"

    lf = tmp_path / "lf.html"
    crlf = tmp_path / "crlf.html"
    lf.write_bytes(body)
    crlf.write_bytes(body.replace(b"\n", b"\r\n"))

    assert stamp(lf) == stamp(crlf), (
        "the stamp is line-ending dependent, so a stamp written on one platform "
        "cannot be verified on another"
    )
