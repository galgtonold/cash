"""Render each FIXTURES entry to docs/_badges/<name>.html.

Usage:
    python scripts/build_badge_examples.py [--out DIR]

Default --out is docs/_badges relative to repo root.

The renderer normally calls ``uuid.uuid4()`` to mint unique checkbox
IDs (so multiple badges on the same page can't collide). For doc
snippets we need byte-stable output across runs, so we monkey-patch
``uuid.uuid4`` with a deterministic counter for the lifetime of this
script. IDs become ``id-0000000001``, ``id-0000000002``, ... — still
unique within and across fixtures, just reproducible.
"""
from __future__ import annotations

import argparse
import itertools
import sys
import uuid
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


class _DeterministicUUID:
    """Stand-in for ``uuid.UUID`` exposing the ``.hex`` attribute the
    renderer reads. We only need ``hex[:10]``-style slicing to work."""

    __slots__ = ("hex",)

    def __init__(self, n: int) -> None:
        self.hex = f"{n:032x}"


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

    for name, metrics in FIXTURES.items():
        view = build_interactive_badge(metrics)
        html = render_html(view)
        (args.out / f"{name}.html").write_text(html, encoding="utf-8")
        print(f"wrote {name}.html ({len(html):,} bytes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
