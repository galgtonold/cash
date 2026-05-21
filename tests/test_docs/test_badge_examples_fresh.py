"""Drift test: re-render badge fixtures and assert they match committed snippets.

If this test fails, run:

    python scripts/build_badge_examples.py

and commit the regenerated docs/_badges/*.html.
"""
from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
BADGES_DIR = REPO_ROOT / "docs" / "_badges"
BUILD_SCRIPT = REPO_ROOT / "scripts" / "build_badge_examples.py"


def test_committed_badges_match_current_render(tmp_path):
    """docs/_badges/*.html must equal what build_badge_examples.py would emit today."""
    out_dir = tmp_path / "_badges"
    out_dir.mkdir()
    result = subprocess.run(
        [sys.executable, str(BUILD_SCRIPT), "--out", str(out_dir)],
        capture_output=True, text=True, check=False,
    )
    assert result.returncode == 0, f"build script failed: {result.stderr}"

    # Only compare files the build script owns (those generated from FIXTURES).
    # Hand-crafted badge HTML in docs/_badges/ (e.g. animated "why-cash" reels
    # or one-off tutorial illustrations) are intentionally excluded.
    committed = {p.name: p.read_text(encoding="utf-8")
                 for p in BADGES_DIR.glob("*.html")}
    rendered = {p.name: p.read_text(encoding="utf-8")
                for p in out_dir.glob("*.html")}

    committed_managed = {k: v for k, v in committed.items() if k in rendered}
    assert committed_managed == rendered, (
        "docs/_badges/*.html is stale. Run "
        "`python scripts/build_badge_examples.py` and commit the result."
    )


def test_every_fixture_renders():
    """Each fixture in scripts/badge_fixtures.py must build without raising."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from scripts.badge_fixtures import FIXTURES
        from cash.notebook.badge_renderer.renderers.html import render_html
        from cash.notebook.badge_renderer.view_builder import build_interactive_badge
    finally:
        sys.path.pop(0)

    assert FIXTURES, "FIXTURES is empty"
    for name, metrics in FIXTURES.items():
        view = build_interactive_badge(metrics)
        html = render_html(view)
        assert html.strip(), f"fixture {name!r} produced empty HTML"
        assert "c3-wrap" in html, f"fixture {name!r} missing badge wrapper class"


def test_render_emits_unique_checkbox_ids():
    """Multi-row badges must have unique rx-* IDs.

    The renderer's per-row <details> toggling depends on every <input id="rx-X">
    being unique within the badge. A deterministic-UUID stub that collides on
    the 10-char slice the renderer uses would produce invalid HTML.
    """
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from cash.notebook.badge_renderer.renderers.html import render_html
        from cash.notebook.badge_renderer.view_builder import build_interactive_badge
        # Patch uuid in the same way the build script does, then build a
        # 3-row fixture that exercises multiple _uid() call sites.
        import scripts.build_badge_examples  # noqa: F401  — applies the patch
    finally:
        sys.path.pop(0)

    metrics = [
        {"status": "RESTORED", "code": "a = 1", "total_time": 0.01, "is_upstream": False},
        {"status": "COMPUTED", "code": "b = a + 1", "total_time": 0.02, "is_upstream": False},
        {"status": "COMPUTED", "code": "c = b + 1", "total_time": 0.03, "is_upstream": False},
    ]
    html = render_html(build_interactive_badge(metrics))
    ids = re.findall(r'id="(rx-[0-9a-f]+)"', html)
    assert len(ids) >= 3, f"expected at least 3 rx-IDs, found {len(ids)}: {ids}"
    assert len(ids) == len(set(ids)), f"duplicate rx-IDs: {ids}"
