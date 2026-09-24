"""The doc-number markers never reach the rendered site.

``scripts/doc_numbers.py`` keeps each derived number between
``<!-- docnum:NAME -->`` and ``<!-- /docnum -->``. In prose those are
invisible HTML comments, but inside inline code or a code block they render
as text. ``mkdocs_hooks.on_page_markdown`` removes them before rendering, and
these tests pin that it removes every marker ``doc_numbers.py`` understands
and keeps the value between them.
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hooks = _load("mkdocs_hooks", REPO_ROOT / "mkdocs_hooks.py")
doc_numbers = _load("doc_numbers", REPO_ROOT / "scripts" / "doc_numbers.py")


@pytest.mark.parametrize(
    "source,expected",
    [
        ("`cash-lib<!-- docnum:version_pin -->~=0.11.0<!-- /docnum -->`", "`cash-lib~=0.11.0`"),
        ("```bash\n# cash <!-- docnum:version -->0.11.0<!-- /docnum -->\n```", "```bash\n# cash 0.11.0\n```"),
        ("About <!--docnum:tests_total-->~8,750<!--  /docnum  --> tests.", "About ~8,750 tests."),
        ("Plain text with <!-- an ordinary comment --> stays.", "Plain text with <!-- an ordinary comment --> stays."),
    ],
)
def test_strip_docnum_markers(source: str, expected: str):
    assert hooks.strip_docnum_markers(source) == expected


def _marked_pages() -> list[Path]:
    pages = sorted((REPO_ROOT / "docs").rglob("*.md"))
    return [p for p in pages if doc_numbers.MARKER.search(p.read_text(encoding="utf-8"))]


@pytest.mark.parametrize("page", _marked_pages(), ids=lambda p: p.relative_to(REPO_ROOT).as_posix())
def test_every_marker_on_the_page_is_removed_and_its_value_kept(page: Path):
    text = page.read_text(encoding="utf-8")
    for match in doc_numbers.MARKER.finditer(text):
        assert hooks.strip_docnum_markers(match.group(0)) == match.group("value")
    assert "docnum" not in hooks.strip_docnum_markers(text)


def test_the_hook_is_registered():
    config = (REPO_ROOT / "mkdocs.yml").read_text(encoding="utf-8")
    assert re.search(r"^hooks:\s*\n\s*-\s*mkdocs_hooks\.py\s*$", config, re.MULTILINE), (
        "mkdocs.yml must list mkdocs_hooks.py under hooks:"
    )
