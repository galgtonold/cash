"""The closed file ``with open(p) as fh:`` leaves bound is stored, and restored as a closed file.

``with open(p) as fh: text = fh.read()``: ``fh`` is one of the statement's
outputs, and a file object does not pickle, so the entry never reached disk
and the failed write was printed in the cell's output on every run.
"""

import copy
import io
import pickle

import pytest

from cash.notebook.cache_status import CacheStatus
from cash.notebook.closed_stream import stored_form
from tests._cell_driver import run_cash_cell
from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S


@pytest.mark.parametrize("mode", ["r", "rb", "w", "ab"])
@pytest.mark.parametrize("copier", [lambda v: pickle.loads(pickle.dumps(v)), copy.deepcopy], ids=["pickled", "copied"])
def test_it_comes_back_as_the_closed_file(tmp_path, mode, copier):
    path = tmp_path / "data.txt"
    path.write_text("hello\n", encoding="latin-1")
    with open(path, mode, **({} if "b" in mode else {"encoding": "latin-1"})) as fh:
        pass
    back = copier(stored_form(fh))
    assert type(back) is type(fh)
    assert (back.name, back.mode, back.closed) == (fh.name, fh.mode, True)
    if isinstance(fh, io.TextIOWrapper):
        assert back.encoding == "latin-1"
    assert path.read_text(encoding="latin-1") == ("hello\n" if mode != "w" else "")


def test_an_open_file_is_left_as_it_is(tmp_path):
    path = tmp_path / "data.txt"
    path.write_text("x", encoding="utf-8")
    with open(path, encoding="utf-8") as fh:
        assert stored_form(fh) is fh


def test_a_hit_restores_the_closed_file(cash_magics, mock_shell, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.txt").write_text("hello world\n", encoding="utf-8")
    cell = f"import time\nwith open('data.txt') as fh:\n    text = fh.read()\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})"
    run_cash_cell(cash_magics, cell)
    del mock_shell.user_ns["fh"], mock_shell.user_ns["text"]
    run_cash_cell(cash_magics, cell)
    statuses = [m.get("status") for m in cash_magics.cash_status("dict")["last_cell"]["statements"]]
    assert statuses[-1] == CacheStatus.RESTORED, statuses
    fh = mock_shell.user_ns["fh"]
    assert isinstance(fh, io.TextIOWrapper) and fh.closed and fh.name == "data.txt"
    assert mock_shell.user_ns["text"] == "hello world\n"
