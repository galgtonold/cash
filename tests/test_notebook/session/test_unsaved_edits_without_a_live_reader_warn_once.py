"""Without a live reader, cash says once that unsaved edits above are invisible.

The upstream check reads the cells above the one being run from the saved
``.ipynb`` unless a live reader (cash's JupyterLab extension, Colab, VS Code's
backup) sends the current ones. In classic Notebook or PyCharm an edit not yet
saved was then checked as its saved code, and the answer was the old one
without a word. ``NOTEBOOK-NO-LIVE-READER`` now says so, once per session, for
the first run a person started (after an idle pause): a Run All or papermill
queues every cell right behind the last, where nobody can have edited.
"""

from __future__ import annotations

import time
import warnings

import pytest

from cash.notebook import live_cells
from cash.notebook import server_discovery as sd

pytestmark = pytest.mark.core


@pytest.fixture
def no_reader(monkeypatch):
    monkeypatch.setattr(sd, "_warned_no_live_reader", False)
    monkeypatch.setattr(sd, "_a_live_reader_can_answer", lambda: False)
    monkeypatch.setattr(sd, "_in_vscode", lambda: False)
    monkeypatch.setattr(sd, "read_notebook_cells", lambda _path=None: sd.CellRead((), "file"))
    return monkeypatch


def _idle(monkeypatch, seconds):
    monkeypatch.setattr(live_cells, "_last_run_end", time.monotonic() - seconds)


def _codes(record):
    return [str(w.message) for w in record if "NOTEBOOK-NO-LIVE-READER" in str(w.message)]


def test_a_run_after_a_pause_warns_once(no_reader):
    _idle(no_reader, 30)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        sd.warn_no_live_reader_once("nb.ipynb")
        sd.warn_no_live_reader_once("nb.ipynb")
    found = _codes(record)
    assert len(found) == 1
    assert "saved notebook file" in found[0]


def test_a_cell_queued_right_behind_the_last_does_not_warn(no_reader):
    _idle(no_reader, 0.01)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        sd.warn_no_live_reader_once("nb.ipynb")
    assert _codes(record) == []
    assert sd._warned_no_live_reader is False


def test_the_first_execution_does_not_warn(no_reader):
    no_reader.setattr(live_cells, "_last_run_end", None)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        sd.warn_no_live_reader_once("nb.ipynb")
    assert _codes(record) == []


@pytest.mark.parametrize("reader", ["extension", "colab", "vscode-backup"])
def test_a_live_read_does_not_warn(no_reader, reader):
    _idle(no_reader, 30)
    no_reader.setattr(sd, "read_notebook_cells", lambda _path=None: sd.CellRead((), reader))
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        sd.warn_no_live_reader_once("nb.ipynb")
    assert _codes(record) == []


def test_vs_code_does_not_warn_while_its_backup_is_clean(no_reader):
    # VS Code writes no backup while nothing is unsaved: the file read is then
    # the current text, not a missed edit.
    _idle(no_reader, 30)
    no_reader.setattr(sd, "_in_vscode", lambda: True)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        sd.warn_no_live_reader_once("nb.ipynb")
    assert _codes(record) == []


def test_the_post_run_hook_marks_when_a_run_ended(monkeypatch):
    monkeypatch.setattr(live_cells, "_last_run_end", None)
    live_cells._on_post_run_cell()
    since = live_cells.seconds_since_last_run()
    assert since is not None and 0 <= since < 5


def test_the_upstream_check_asks_for_cells_above(no_reader, cash_magics):
    from tests._cell_driver import run_cash_cell

    _idle(no_reader, 30)
    with warnings.catch_warnings(record=True) as record:
        warnings.simplefilter("always")
        run_cash_cell(cash_magics, "y = x + 1", cells=["x = 1", "y = x + 1"])
    assert len(_codes(record)) == 1
