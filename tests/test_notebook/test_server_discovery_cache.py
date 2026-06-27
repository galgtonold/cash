"""Memoization + freshness for notebook-file reads in server_discovery.

The upstream checker reads the notebook file many times per cell run (once in
the magic, twice in the checker). Re-parsing the same unchanged file on every
read is wasteful, and each read paid a ``_wait_for_notebook_save`` sleep — so a
single ``run_all`` could spend >1s just re-reading the same file.

These tests pin the contract:
  * reads of an UNCHANGED file are memoized (parsed once), and
  * the cache invalidates the moment the file changes (mtime/size), so a quick
    edit-then-run never serves stale cell sources.
"""
from __future__ import annotations

import json as _json
import os
import time

import pytest

from cash.notebook import server_discovery as sd


def _write_nb(path, sources: list[str]) -> None:
    cells = [{"cell_type": "code", "source": s} for s in sources]
    path.write_text(
        _json.dumps({"cells": cells, "metadata": {}, "nbformat": 4, "nbformat_minor": 5}),
        encoding="utf-8",
    )


def _age(path, seconds: float) -> None:
    """Set the file's mtime to *seconds* in the past."""
    t = time.time() - seconds
    os.utime(path, (t, t))


@pytest.fixture(autouse=True)
def _clear_cache():
    sd.invalidate_notebook_cells_cache()
    yield
    sd.invalidate_notebook_cells_cache()


@pytest.fixture
def _count_parses(monkeypatch):
    """Count json.load calls inside server_discovery without affecting other json use."""
    counter = {"n": 0}
    real_load = _json.load

    class _Shim:
        @staticmethod
        def load(f):
            counter["n"] += 1
            return real_load(f)

    monkeypatch.setattr(sd, "json", _Shim)
    return counter


def test_repeated_reads_parse_once(tmp_path, _count_parses):
    nb = tmp_path / "n.ipynb"
    _write_nb(nb, ["a = 1", "b = 2"])
    _age(nb, 100)  # old file → no save-wait

    results = [sd.get_notebook_cells(str(nb)) for _ in range(5)]

    assert all(r == ["a = 1", "b = 2"] for r in results)
    assert _count_parses["n"] == 1, f"expected 1 parse for 5 reads, got {_count_parses['n']}"


def test_cache_invalidates_on_edit(tmp_path, _count_parses):
    nb = tmp_path / "n.ipynb"
    _write_nb(nb, ["a = 1"])
    _age(nb, 100)
    assert sd.get_notebook_cells(str(nb)) == ["a = 1"]

    # Edit the file: new content, fresh mtime.
    _write_nb(nb, ["a = 999"])
    _age(nb, 100)  # keep it "old" so the wait doesn't interfere with the assertion

    assert sd.get_notebook_cells(str(nb)) == ["a = 999"], "served STALE cells after edit"
    assert _count_parses["n"] == 2, "edit should force a fresh parse"


def test_include_ids_variant_cached_separately(tmp_path):
    nb = tmp_path / "n.ipynb"
    _write_nb(nb, ["x = 1"])
    _age(nb, 100)

    plain = sd.get_notebook_cells(str(nb))
    with_ids = sd.get_notebook_cells_with_ids(str(nb))

    assert plain == ["x = 1"]
    assert [code for _id, code in with_ids] == ["x = 1"]
