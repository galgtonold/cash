"""Budgets for what a badge costs a saved notebook.

Executing the ten-cell feature tour used to produce a 528.6 KB .ipynb of which
354.9 KB -- 67% -- was twelve identical copies of the badge stylesheet.

Two invariants keep that from coming back. The second is the subtle one: file
size scales with CELLS, not updates, because `display(..., update=True)`
mutates one stored output in place. A code path that renders without a
`display_id` falls through to `display(HTML(html))` and creates a fresh output
every time, which would quietly multiply a notebook's size.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
NOTEBOOK = ROOT / "examples" / "try_cash_binder.ipynb"


@pytest.fixture(scope="module")
def executed(tmp_path_factory):
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("nbclient")
    from nbclient import NotebookClient

    work = tmp_path_factory.mktemp("badgebudget")
    nb = nbformat.read(NOTEBOOK, as_version=4)
    NotebookClient(nb, timeout=600, kernel_name="python3",
                   resources={"metadata": {"path": str(work)}}).execute()
    return nb


def test_one_badge_output_per_cell(executed):
    """Updates replace, they do not accumulate."""
    for i, cell in enumerate(executed.cells):
        if cell.cell_type != "code":
            continue
        badges = 0
        for out in cell.get("outputs", []):
            html = (out.get("data") or {}).get("text/html", "")
            if isinstance(html, list):
                html = "".join(html)
            if "c3-card" in html:
                badges += 1
        assert badges <= 1, f"cell {i} stored {badges} badge outputs, expected at most 1"


def test_stylesheet_is_a_minority_of_the_saved_notebook(executed):
    import nbformat

    text = nbformat.writes(executed)
    css = sum(len(s) for s in re.findall(r"<style>.*?</style>", text, re.S))
    share = css / len(text)
    assert share < 0.60, (
        f"the badge stylesheet is {share:.0%} of the notebook "
        f"({css/1024:.0f} KB of {len(text)/1024:.0f} KB); it was 67% before "
        f"minification and should now be about 56%"
    )
