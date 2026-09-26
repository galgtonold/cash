"""The build-time rewrites in ``mkdocs_hooks.py``.

The published page and the Markdown source differ on purpose: the source
keeps what the docs tests read (``# test:inject:`` lines, the "Applies to"
box), and the hook turns it into what a reader needs. These tests pin each
rewrite, and that it keeps working on the pages as they are.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

from tests.docs._harness import _apply_inject_comments

REPO_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("mkdocs_hooks", REPO_ROOT / "mkdocs_hooks.py")
hooks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hooks)


def _pages() -> list[Path]:
    return [p for p in sorted((REPO_ROOT / "docs").rglob("*.md")) if "superpowers" not in p.parts]


# --- "# test:inject:" lines -------------------------------------------------


def test_inject_lines_are_dropped_and_the_rest_kept():
    source = "```python\nimport cash\n    # test:inject: x = 1\nprint(x)\n# a comment\n```\n"
    assert hooks.strip_test_injects(source) == "```python\nimport cash\nprint(x)\n# a comment\n```\n"


def test_no_line_the_harness_injects_reaches_the_site():
    """Every line the harness turns into code is one the hook removes."""
    kept = [
        f"{page.relative_to(REPO_ROOT).as_posix()}: {line.strip()}"
        for page in _pages()
        for line in page.read_text(encoding="utf-8").splitlines()
        if _apply_inject_comments(line) != line and hooks.strip_test_injects(line + "\n")
    ]
    assert not kept, "\n".join(kept)
