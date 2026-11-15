"""What a process works out before its cache folder exists is written
once the folder is there (`cash.analysis.scan_store`).

A first call is a miss, and it is analysed before the result that makes
the cache folder is stored: the store, which never makes a folder of its
own, kept nothing, so the second process paid for the whole analysis
again (a Cython module's parse, about a second each).
"""

from __future__ import annotations

import pytest

from cash._active import default_cash, set_default_cash
from cash.analysis import scan_store


@pytest.fixture(autouse=True)
def _settings_from_the_environment():
    """No default instance an earlier test left behind: its cache dir would
    stand in for the ``CASH_CACHE_DIR`` each test here sets."""
    earlier = default_cash()
    set_default_cash(None)
    yield
    set_default_cash(earlier)


def test_a_record_saved_before_the_folder_exists_is_written_once_it_does(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setenv("CASH_CACHE_DIR", str(cache))
    monkeypatch.delenv("CASH_ANALYSIS_STORE", raising=False)
    monkeypatch.setattr(scan_store, "_PENDING", {})
    assert scan_store.store_dir() is None

    scan_store.save_record("probe-kind", "/some/file", "d1", {"answer": 1})
    names = scan_store.scan("probe-scan", "/some/module.py", "x = 1\n", lambda text: frozenset({"x"}))
    assert names == frozenset({"x"})
    assert not cache.exists(), "the store made a cache folder of its own"
    assert scan_store.load_record("probe-kind", "/some/file", "d1") is None

    cache.mkdir()
    scan_store.flush()
    assert scan_store.load_record("probe-kind", "/some/file", "d1")["answer"] == 1
    calls = []
    again = scan_store.scan("probe-scan", "/some/module.py", "x = 1\n", lambda text: calls.append(text))
    assert again == frozenset({"x"}) and not calls, "the scan held for the folder was not written"


def test_a_held_record_is_written_with_the_next_one(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    monkeypatch.setenv("CASH_CACHE_DIR", str(cache))
    monkeypatch.delenv("CASH_ANALYSIS_STORE", raising=False)
    monkeypatch.setattr(scan_store, "_PENDING", {})
    scan_store.save_record("probe-kind", "/a", "d", {"answer": "a"})
    cache.mkdir()
    scan_store.save_record("probe-kind", "/b", "d", {"answer": "b"})
    assert scan_store.load_record("probe-kind", "/a", "d")["answer"] == "a"
    assert not scan_store._PENDING


def test_nothing_is_held_with_the_store_off(tmp_path, monkeypatch):
    monkeypatch.setenv("CASH_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("CASH_ANALYSIS_STORE", "0")
    monkeypatch.setattr(scan_store, "_PENDING", {})
    scan_store.save_record("probe-kind", "/a", "d", {"answer": "a"})
    assert not scan_store._PENDING
