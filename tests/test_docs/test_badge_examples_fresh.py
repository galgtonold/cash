"""Drift test: re-render badge fixtures and assert they match committed snippets.

If this test fails, run:

    python scripts/build_badge_examples.py

and commit the regenerated docs/_badges/*.html.
"""
from __future__ import annotations

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

    committed = {p.name: p.read_text(encoding="utf-8")
                 for p in BADGES_DIR.glob("*.html")}
    rendered = {p.name: p.read_text(encoding="utf-8")
                for p in out_dir.glob("*.html")}

    assert committed == rendered, (
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
