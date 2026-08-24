"""Render docs/_brand/social-card.html to the PNG GitHub shows as a link preview.

The card is what renders when the repo is posted to Show HN, Reddit, Discourse
or Slack. It is set once in the repository's Settings -> Social preview, and is
otherwise invisible from inside the project -- which is exactly the kind of
asset that goes stale without anyone noticing, because nothing you look at
day to day is wrong.

Hence the same chain of custody the badge images use: the PNG is a build
product of committed HTML, and ``tests/test_docs/test_brand_assets_fresh.py``
fails if the HTML moves without the picture being rebuilt.

The stamp function is IMPORTED from build_badge_images rather than copied. It
hashes newline-normalized bytes, because the HTML is CRLF in a Windows working
tree and LF everywhere else -- a raw-byte hash records the checkout rather than
the content, and a stamp written on one platform then fails to verify on every
other. That is not hypothetical: it is precisely what a duplicated stamp
implementation did to CI, and duplicating it again is how it would come back.

Needs playwright with its bundled chromium (``pip install playwright &&
playwright install chromium``). Only maintainers regenerating brand art need
it -- the PNG is committed, so a normal checkout, test run and docs build
never touch playwright.

Usage:  python scripts/build_social_card.py [--check]
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
BRAND_DIR = REPO_ROOT / "docs" / "_brand"
NAME = "social-card"

# GitHub renders the social preview at 1280x640. Rendering at 2x keeps it crisp
# where the card is shown large (Slack unfurls, the Settings preview) and
# downsamples cleanly everywhere else.
WIDTH, HEIGHT, SCALE = 1280, 640, 2


def _stamp_fn():
    """Borrow the badge builder's stamp so the two cannot disagree.

    Loaded by path rather than imported as a module: scripts/ is not on
    sys.path for a plain ``python scripts/...`` invocation.
    """
    path = REPO_ROOT / "scripts" / "build_badge_images.py"
    spec = importlib.util.spec_from_file_location("build_badge_images", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod._stamp


def render() -> int:
    html = BRAND_DIR / f"{NAME}.html"
    if not html.exists():
        print(f"missing {html}", file=sys.stderr)
        return 1

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "playwright is not installed. It is only needed to regenerate the\n"
            "social card, which is committed:\n"
            "    pip install playwright && playwright install chromium",
            file=sys.stderr,
        )
        return 2

    out = BRAND_DIR / f"{NAME}.png"
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(
            viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=SCALE
        )
        page.goto(html.as_uri())
        # No webfonts to wait on -- the design system is system-stack only --
        # but give the gradients a frame to settle before capturing.
        page.wait_for_timeout(250)
        page.screenshot(path=str(out))
        browser.close()

    (BRAND_DIR / f"{NAME}.png.stamp").write_text(_stamp_fn()(html), encoding="utf-8")
    print(f"wrote {out.name} ({out.stat().st_size:,} bytes) at {WIDTH}x{HEIGHT}@{SCALE}x")
    return 0


def check() -> int:
    """Report whether the committed PNG is current, without needing playwright."""
    html = BRAND_DIR / f"{NAME}.html"
    stamp = BRAND_DIR / f"{NAME}.png.stamp"
    if not stamp.exists():
        print(f"{NAME}.png has no .stamp; run this script without --check")
        return 1
    expected = _stamp_fn()(html)
    actual = stamp.read_text(encoding="utf-8").strip()
    if actual != expected:
        print(f"stale: {NAME}.html has changed since the PNG was rendered "
              f"({actual} != {expected})")
        return 1
    print(f"{NAME}.png is current ({expected})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verify the committed PNG is current; no playwright needed")
    args = ap.parse_args()
    return check() if args.check else render()


if __name__ == "__main__":
    raise SystemExit(main())
