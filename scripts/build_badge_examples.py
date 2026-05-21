"""Render each FIXTURES entry to docs/_badges/<name>.html.

Usage:
    python scripts/build_badge_examples.py [--out DIR]

Default --out is docs/_badges relative to repo root.

Each output file is a STANDALONE HTML document (DOCTYPE + html + head +
body) so the badge can be embedded in the docs via ``<iframe src="…">``
and rendered in an isolated context that mkdocs-material's global CSS
cannot reach. The wrapper also includes a tiny postMessage resize
script so the parent page can size the iframe to the badge's content
height (initially and after the badge expands).

The renderer normally calls ``uuid.uuid4()`` to mint unique checkbox
IDs. For doc snippets we need byte-stable output across runs, so we
monkey-patch ``uuid.uuid4`` with a deterministic counter for the
lifetime of this script. IDs become ``id-0000000001``,
``id-0000000002``, ... — still unique within and across fixtures, just
reproducible.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


class _DeterministicUUID:
    """Stand-in for ``uuid.UUID`` exposing the ``.hex`` attribute the
    renderer reads. We only need ``hex[:10]``-style slicing to work."""

    __slots__ = ("hex",)

    def __init__(self, n: int) -> None:
        # Put the counter in the LEADING hex digits so the renderer's
        # ``.hex[:10]`` slice is unique per call. A 32-char hex string
        # matches the standard ``uuid.UUID.hex`` length.
        self.hex = f"{n:010x}" + "0" * 22


_uuid_counter = itertools.count(1)


def _deterministic_uuid4() -> _DeterministicUUID:
    return _DeterministicUUID(next(_uuid_counter))


# Patch BEFORE importing the renderer so the module-level ``import uuid``
# inside html.py resolves to our stub when it later calls ``uuid.uuid4()``.
uuid.uuid4 = _deterministic_uuid4  # type: ignore[assignment]

from cash.notebook.badge_renderer.renderers.html import render_html  # noqa: E402
from cash.notebook.badge_renderer.view_builder import build_interactive_badge  # noqa: E402
from scripts.badge_fixtures import FIXTURES  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO_ROOT / "docs" / "_badges",
        help="Output directory for rendered HTML snippets.",
    )
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Per-fixture kwargs forwarded to build_interactive_badge.
    fixture_kwargs: dict[str, dict[str, Any]] = {
        "anatomy_hero": {
            "timing_breakdown": {
                "upstream_check": 0.011,
                "badge_init": 0.008,
                "badge_progress": 0.005,
            },
            "cell_total_time": 0.421,
        },
    }

    for name, metrics in FIXTURES.items():
        view = build_interactive_badge(metrics, **fixture_kwargs.get(name, {}))
        badge_html = render_html(view)
        doc = _wrap_standalone(badge_html, title=name)
        (args.out / f"{name}.html").write_text(doc, encoding="utf-8")
        print(f"wrote {name}.html ({len(doc):,} bytes)")

    return 0


# A standalone HTML wrapper that:
# - resets browser default margins so the badge sits flush in its iframe;
# - posts a 'cash-badge-resize' message to the parent on load, on every
#   <details> toggle inside the badge, and on any size change, so the
#   parent can resize the hosting iframe to fit.
_WRAPPER = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Cash badge: {title}</title>
<style>
  html, body {{ margin: 0; padding: 0; background: transparent; }}
  body {{ padding: 1px; }}  /* avoid clipping the card's 1px outer border */
</style>
</head>
<body>
{body}
<script>
(function () {{
  if (window.parent === window) return;
  function sendSize() {{
    var h = Math.ceil(document.documentElement.getBoundingClientRect().height);
    window.parent.postMessage({{ type: "cash-badge-resize", height: h }}, "*");
  }}
  // Send on load, then re-send a few times with backoff so the parent's
  // resize listener still receives a message even if the iframe document
  // was cached and fired its load event before the parent's script ran.
  function burstSend() {{
    sendSize();
    setTimeout(sendSize, 50);
    setTimeout(sendSize, 200);
    setTimeout(sendSize, 1000);
  }}
  if (document.readyState === "complete") burstSend();
  else window.addEventListener("load", burstSend);
  // Re-send on every <details> open/close so the iframe grows/shrinks
  // when the badge expands.
  document.addEventListener("toggle", sendSize, true);
  // Catch any other layout change (font load, reflow).
  if (typeof ResizeObserver !== "undefined") {{
    new ResizeObserver(sendSize).observe(document.documentElement);
  }}
}})();
</script>
</body>
</html>
"""


def _wrap_standalone(badge_html: str, *, title: str) -> str:
    """Wrap a badge fragment in a self-contained HTML document."""
    return _WRAPPER.format(title=title, body=badge_html)


if __name__ == "__main__":
    raise SystemExit(main())
