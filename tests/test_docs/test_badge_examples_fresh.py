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


def test_render_is_independent_of_ambient_uuid_consumption(tmp_path):
    """The snapshots must not depend on UUIDs minted earlier in the process.

    The build script stubs ``uuid.uuid4`` with a counter to get byte-stable
    output. That counter used to be process-global and never reset, so every
    fixture's checkbox IDs encoded how many ``uuid4()`` calls had already
    happened — including calls made while importing unrelated packages.

    That is exactly how this broke: the machine the fixtures were generated on
    had ``opentelemetry`` in its user site-packages, which mints 4 UUIDs at
    import time. Every committed file was offset by 4 from what a clean
    environment renders, so the drift test above passed on that one machine and
    failed on all 15 CI jobs.

    Burning a few UUIDs before the script runs simulates any such package.
    Output must be unchanged.
    """
    out_dir = tmp_path / "_badges"
    out_dir.mkdir()
    preamble = (
        "import sys, uuid, runpy\n"
        "[uuid.uuid4() for _ in range(7)]\n"          # stand-in for opentelemetry et al.
        f"sys.argv = ['build_badge_examples.py', '--out', {str(out_dir)!r}]\n"
        "try:\n"
        f"    runpy.run_path({str(BUILD_SCRIPT)!r}, run_name='__main__')\n"
        "except SystemExit as e:\n"
        "    sys.exit(e.code)\n"
    )
    result = subprocess.run([sys.executable, "-c", preamble],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"build script failed: {result.stderr}"

    rendered = {p.name: p.read_text(encoding="utf-8") for p in out_dir.glob("*.html")}
    assert rendered, "build script produced no output"
    committed = {p.name: p.read_text(encoding="utf-8")
                 for p in BADGES_DIR.glob("*.html") if p.name in rendered}
    assert committed == rendered, (
        "Badge output shifted when unrelated code consumed uuid4() first. The "
        "per-fixture counter reset in scripts/build_badge_examples.py is gone "
        "or ineffective, so the snapshots depend on whatever happens to be "
        "installed on the machine that generated them."
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


def test_render_emits_unique_checkbox_ids(monkeypatch):
    """Multi-row badges must have unique rx-* IDs.

    The renderer's per-row <details> toggling depends on every <input id="rx-X">
    being unique within the badge. A deterministic-UUID stub that collides on
    the 10-char slice the renderer uses would produce invalid HTML.

    We patch ``uuid.uuid4`` with a low-entropy sequential stub (so the
    assertion is reproducible) but do it via ``monkeypatch`` so it is restored
    after the test. Importing ``scripts.build_badge_examples`` for its patch
    side-effect (as this test used to) leaked the stub process-wide and left
    every later badge render on the same xdist worker using deterministic IDs -
    a cross-test contamination flake.
    """
    import itertools
    import uuid

    class _DetUUID:
        __slots__ = ("hex",)

        def __init__(self, n: int) -> None:
            self.hex = f"{n:010x}" + "0" * 22

    _counter = itertools.count(1)
    monkeypatch.setattr(uuid, "uuid4", lambda: _DetUUID(next(_counter)))

    sys.path.insert(0, str(REPO_ROOT))
    try:
        from cash.notebook.badge_renderer.renderers.html import render_html
        from cash.notebook.badge_renderer.view_builder import build_interactive_badge
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
