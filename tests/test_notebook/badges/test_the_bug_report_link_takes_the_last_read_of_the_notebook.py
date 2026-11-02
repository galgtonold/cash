"""The bug report link of a badge takes the notebook's cells from the last read of them.

Each badge carries a link to a pre-filled bug report holding the notebook's
cells, and building it read the notebook again, through every reader: on a
Windows machine, 2 s a badge, before a cell that took 0.01 s. The check
before the cell read the cells already; the link takes that read.
"""

from __future__ import annotations

import types

import pytest

from cash.notebook import server_discovery
from cash.notebook.ipython import badges
from cash.notebook.ipython.badges import BadgePresenter
from cash.notebook.server_discovery import NotebookCell


@pytest.fixture
def reads(monkeypatch) -> list[str | None]:
    """The reads of the notebook's cells that reached a reader."""
    asked: list[str | None] = []
    readers = server_discovery._readers

    def saved_file(notebook_path):
        asked.append(notebook_path)
        return (NotebookCell("c1", "import pandas as pd"), NotebookCell("c2", "df.shape"))

    monkeypatch.setattr(server_discovery, "_try_extension_cells", lambda: None)
    monkeypatch.setattr(readers, "colab", lambda: None)
    monkeypatch.setattr(readers, "vscode_backup", lambda notebook_path: None)
    monkeypatch.setattr(readers, "saved_file", saved_file)
    monkeypatch.setattr(server_discovery, "get_notebook_path", lambda: "/w/nb.ipynb")
    if hasattr(readers, "last"):
        monkeypatch.setattr(readers, "last", None)
    return asked


def _context() -> dict:
    shell = types.SimpleNamespace(execution_count=7, history_manager=None, user_ns={"In": ["df.shape"]})
    return BadgePresenter(shell, None)._collect_bug_report_context()


def test_the_link_takes_the_cells_the_check_read(reads):
    server_discovery.read_notebook_cells("/w/nb.ipynb")  # the check before the cell
    reads.clear()

    assert _context()["notebook_source"] == ["import pandas as pd", "df.shape"]
    assert reads == []


def test_before_any_read_the_link_reads_the_notebook(reads, monkeypatch):
    monkeypatch.setattr(badges, "get_notebook_cells", server_discovery.get_notebook_cells)

    assert _context()["notebook_source"] == ["import pandas as pd", "df.shape"]
    assert len(reads) == 1
