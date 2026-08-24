"""Render docs/_brand/*.html to the brand PNGs GitHub and the README show.

Three assets from two sources:

    social-card.png          1280x640, GitHub Settings -> Social preview
    readme-header-light.png  1280x280, top of README.md
    readme-header-dark.png   the same source, rendered dark

The header is one source rendered twice under different prefers-color-scheme,
so the two variants cannot drift apart in copy or layout -- only in colour.
GitHub chooses between them with <picture media="(prefers-color-scheme: dark)">.

These are the kind of asset that goes stale without anyone noticing. The social
card is set once in Settings and never seen from inside the project; the header
is seen constantly and therefore stops being read. Nothing you look at day to
day would be wrong.

Hence the same chain of custody the badge images use: each PNG is a build
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
it -- the PNGs are committed, so a normal checkout, test run and docs build
never touch playwright.

Usage:  python scripts/build_brand_assets.py [--check]
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
BRAND_DIR = REPO_ROOT / "docs" / "_brand"

#: (source stem, output stem, width, height, scale, colour scheme)
#:
#: Scale is per-asset, not global. The README header renders at 2x so it stays
#: crisp on a HiDPI screen, where GitHub displays it well below its natural
#: width. The social card renders at 1x: GitHub documents 1280x640 as the size
#: for best display and caps the upload at 1MB, and a 2x card came out
#: 2560x1280 at 822KB -- inside the stated limit, but it would not take. Match
#: the documented size exactly rather than argue with the uploader.
ASSETS = [
    ("social-card",   "social-card",         1280, 640, 1, "light"),
    ("readme-header", "readme-header-light", 1280, 280, 2, "light"),
    ("readme-header", "readme-header-dark",  1280, 280, 2, "dark"),
]


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
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print(
            "playwright is not installed. It is only needed to regenerate the "
            "brand assets, which are committed:\n"
            "    pip install playwright && playwright install chromium",
            file=sys.stderr,
        )
        return 2

    stamp = _stamp_fn()
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        try:
            for src, out_stem, width, height, scale, scheme in ASSETS:
                html = BRAND_DIR / f"{src}.html"
                if not html.exists():
                    print(f"missing {html}", file=sys.stderr)
                    return 1
                page = browser.new_page(
                    viewport={"width": width, "height": height},
                    device_scale_factor=scale,
                    color_scheme=scheme,
                )
                page.goto(html.as_uri())
                # No webfonts to wait on -- the design system is system-stack
                # only -- but let the gradients settle before capturing.
                page.wait_for_timeout(250)
                out = BRAND_DIR / f"{out_stem}.png"
                page.screenshot(path=str(out))
                page.close()
                # The stamp names the SOURCE, so a variant rendered from a
                # shared html is held to that html.
                (BRAND_DIR / f"{out_stem}.png.stamp").write_text(
                    stamp(html), encoding="utf-8")
                print(f"wrote {out.name} ({out.stat().st_size:,} bytes) "
                      f"{width * scale}x{height * scale} "
                      f"({width}x{height}@{scale}x, {scheme})")
        finally:
            browser.close()
    return 0


def check() -> int:
    """Report whether the committed PNGs are current, without needing playwright."""
    stamp = _stamp_fn()
    stale = []
    for src, out_stem, *_ in ASSETS:
        html = BRAND_DIR / f"{src}.html"
        stamp_path = BRAND_DIR / f"{out_stem}.png.stamp"
        if not stamp_path.exists():
            stale.append(f"{out_stem}.png has no .stamp")
            continue
        expected = stamp(html)
        actual = stamp_path.read_text(encoding="utf-8").strip()
        if actual != expected:
            stale.append(f"{out_stem}.png is stale ({actual} != {expected})")
    if stale:
        for line in stale:
            print(line)
        print("run: python scripts/build_brand_assets.py")
        return 1
    print(f"all {len(ASSETS)} brand assets current")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true",
                    help="verify the committed PNGs are current; no playwright needed")
    args = ap.parse_args()
    return check() if args.check else render()


if __name__ == "__main__":
    raise SystemExit(main())
