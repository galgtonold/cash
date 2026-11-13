"""The saved notebook is read again exactly when it may have changed.

The upstream check reads the ``.ipynb`` on every cell; a read of an
unchanged file is served from the last one, with no save-settle wait and
no parse. It was keyed on ``(mtime_ns, size)`` alone: a same-size rewrite
that left the mtime as it was (``cp -p``, an editor restoring it, a second
save inside one timestamp tick) was served the old cells. The key now
holds the file's identity and change time too, and a read taken inside the
timestamp tick of the file's last write is not reused, as a later write in
that tick would leave every stamp the same.
"""

from __future__ import annotations

import json
import os
import time
import types
from pathlib import Path

from cash.notebook import server_discovery as sd


def _write(path: Path, code: str) -> None:
    nb = {
        "cells": [{"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": code}],
        "metadata": {},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(nb), encoding="utf-8")


def _counting(monkeypatch) -> dict[str, int]:
    seen = {"sleep": 0, "load": 0}
    real_sleep, real_load = sd._time.sleep, sd.json.loads

    def sleep(seconds):
        seen["sleep"] += 1
        real_sleep(seconds)

    def load(text):
        seen["load"] += 1
        return real_load(text)

    monkeypatch.setattr(sd._time, "sleep", sleep)
    monkeypatch.setattr(sd.json, "loads", load)
    return seen


def test_an_unchanged_settled_file_is_neither_waited_on_nor_read_again(tmp_path, monkeypatch):
    nb = tmp_path / "nb.ipynb"
    _write(nb, "x = 1")
    old = time.time_ns() - 10 * 1_000_000_000
    os.utime(nb, ns=(old, old))
    readers = sd.NotebookCellReaders()
    assert [c.code for c in readers.saved_file(str(nb))] == ["x = 1"]
    seen = _counting(monkeypatch)
    for _ in range(3):
        assert [c.code for c in readers.saved_file(str(nb))] == ["x = 1"]
    assert seen == {"sleep": 0, "load": 0}


def test_a_same_size_rewrite_with_its_mtime_put_back_is_read_again(tmp_path):
    nb = tmp_path / "nb.ipynb"
    _write(nb, "x = 1")
    old = time.time_ns() - 10 * 1_000_000_000
    os.utime(nb, ns=(old, old))
    readers = sd.NotebookCellReaders()
    assert [c.code for c in readers.saved_file(str(nb))] == ["x = 1"]
    _write(nb, "x = 2")  # same size
    os.utime(nb, ns=(old, old))
    assert [c.code for c in readers.saved_file(str(nb))] == ["x = 2"]


def test_where_ctime_is_the_creation_time_a_same_size_rewrite_is_read_again(tmp_path, monkeypatch):
    """Windows: ``st_ctime`` is when the file was made, which a rewrite in
    place leaves as it was, so the stamps alone served the old cells. The
    bytes are compared there instead."""
    nb = tmp_path / "nb.ipynb"
    _write(nb, "x = 1")
    old = time.time_ns() - 10 * 1_000_000_000
    os.utime(nb, ns=(old, old))
    real_stat = os.stat

    def created_at_old(path, *args, **kwargs):
        st = real_stat(path, *args, **kwargs)
        if Path(path) != nb:
            return st
        fields = {f: getattr(st, f) for f in dir(st) if f.startswith("st_")}
        return types.SimpleNamespace(**{**fields, "st_ctime": old / 1e9, "st_ctime_ns": old})

    monkeypatch.setattr(sd.os, "stat", created_at_old)
    monkeypatch.setattr(sd, "_STAT_HAS_CHANGE_TIME", False)
    readers = sd.NotebookCellReaders()
    assert [c.code for c in readers.saved_file(str(nb))] == ["x = 1"]
    seen = _counting(monkeypatch)
    assert [c.code for c in readers.saved_file(str(nb))] == ["x = 1"]
    assert seen == {"sleep": 0, "load": 0}, "control: an unchanged file is served"

    _write(nb, "x = 2")  # same size
    os.utime(nb, ns=(old, old))
    assert sd._stat_sig(nb) == readers._file_cells[str(nb)][0], "control: the stamps did not move"
    assert [c.code for c in readers.saved_file(str(nb))] == ["x = 2"]


def test_a_read_inside_the_tick_of_the_last_write_is_not_reused(tmp_path, monkeypatch):
    """A filesystem stamping whole seconds: a write later in the same
    second leaves size, mtime and ctime as they were."""
    nb = tmp_path / "nb.ipynb"
    _write(nb, "x = 1")
    real = os.stat(nb)
    tick = (time.time_ns() // 1_000_000_000) * 1_000_000_000

    class Stat:
        st_ino = real.st_ino
        st_dev = real.st_dev
        st_size = real.st_size
        st_mtime_ns = st_ctime_ns = tick
        st_mtime = st_ctime = tick / 1e9

    real_stat = os.stat
    monkeypatch.setattr(sd.os, "stat", lambda p, *a, **k: Stat() if Path(p) == nb else real_stat(p, *a, **k))
    readers = sd.NotebookCellReaders()
    assert [c.code for c in readers.saved_file(str(nb))] == ["x = 1"]
    _write(nb, "x = 2")  # same size, same (faked) stamps
    assert [c.code for c in readers.saved_file(str(nb))] == ["x = 2"]
