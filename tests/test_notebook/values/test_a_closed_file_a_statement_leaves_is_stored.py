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


def test_the_disk_write_gets_the_stored_form_not_the_ram_copy(cash_magics, cash_instance, tmp_path, monkeypatch):
    """The RAM tier's copy of the stored form is the closed file again, which
    does not pickle; handed to the disk write in place of the payload, it made
    the entry miss the disk (and the cell run again after a restart)."""
    from cash.backends.memory_backend import NO_PRIVATE_COPY

    monkeypatch.chdir(tmp_path)
    (tmp_path / "data.txt").write_text("hello world\n", encoding="utf-8")
    written = []
    real_set = cash_instance.backend.set

    def recording_set(key, value, metadata=None, *args, **kwargs):
        written.append((value, dict(metadata or {})))
        return real_set(key, value, metadata, *args, **kwargs)

    monkeypatch.setattr(cash_instance.backend, "set", recording_set)
    run_cash_cell(
        cash_magics,
        f"import time\nwith open('data.txt') as fh:\n    text = fh.read()\n    time.sleep({ABOVE_PERSISTENCE_FLOOR_S})",
    )
    stored = [(v, m) for v, m in written if isinstance(v, dict) and "fh" in v.get("variables", {})]
    assert stored, written
    value, metadata = stored[-1]
    assert metadata.get(NO_PRIVATE_COPY) is True
    pickle.dumps(value)  # what the disk is handed pickles
    # Positive control: a statement leaving no closed file is not marked.
    run_cash_cell(cash_magics, f"y = len(text)\ntime.sleep({ABOVE_PERSISTENCE_FLOOR_S})")
    assert not written[-1][1].get(NO_PRIVATE_COPY)


def test_a_later_persist_writes_the_stored_form_and_restores_the_closed_file(tmp_path):
    """The end-of-cell pass writes an entry only RAM holds (`persist_from_memory`)
    from the RAM tier's own copy. That copy of a stored closed file was the
    closed file again, which does not pickle: the write failed, and the cell
    ran again after a restart."""
    from cash.backends.file_backend import FileBackend
    from cash.backends.memory_backend import NO_PRIVATE_COPY, InMemoryBackend
    from cash.backends.tiered_backend import TieredBackend

    path = tmp_path / "data.txt"
    path.write_text("hello\n", encoding="utf-8")
    with open(path, encoding="utf-8") as fh:
        text = fh.read()
    tiers = TieredBackend([InMemoryBackend(), FileBackend(str(tmp_path / "cache"), flush_interval=0)])
    cheap = {  # a notebook value cheap to run itself: kept in RAM, for the end-of-cell pass
        "execution_time": 0.02,
        "cost_model_family": "_GENERIC",
        "cost_model_type_name": "dict",
        "cost_model_size_bytes": 1000,
        NO_PRIVATE_COPY: True,
    }
    tiers.set("k", {"variables": {"fh": stored_form(fh), "text": text}}, cheap)
    assert tiers.persist_from_memory("k", rebuild_seconds=5.0)
    tiers.shutdown()
    _meta, back = TieredBackend([InMemoryBackend(), FileBackend(str(tmp_path / "cache"))]).get("k")
    restored = back["variables"]["fh"]
    assert type(restored) is type(fh) and restored.closed and restored.name == fh.name
    assert back["variables"]["text"] == "hello\n"
    _meta, from_ram = tiers.backends[0].get("k")
    assert type(from_ram["variables"]["fh"]) is type(fh)  # a RAM hit hands the closed file back
