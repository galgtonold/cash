"""``f.cache_clear()`` reaches running processes, and says what it could not remove.

decorator.md lists ``f.cache_clear()`` and ``cash clear --function`` side by
side, and promises that a clear reaches processes still running. Only the CLI
moved the disk tier's generation, which is what a running process watches: after
``f.cache_clear()`` elsewhere, a server kept serving the cleared result from its
RAM tier indefinitely.

And an entry file another process held open (on Windows: a reader, a virus
scanner) was skipped without a word -- after its bookkeeping had been dropped --
so the next call was a hit on the entry that was meant to be gone.

Two ``Cash`` instances over one folder stand in for two processes: each has its
own RAM tier and its own clear watcher.
"""

from __future__ import annotations

import os
import warnings

import pytest

from cash import Cash
from cash.backends import clear_watch, file_eviction
from cash.exceptions import CashCacheStoreFailedWarning


@pytest.fixture(autouse=True)
def _watch_every_read(monkeypatch):
    monkeypatch.setattr(clear_watch.ClearWatcher, "CHECK_EVERY", 0.0)


def _pair(tmp_path):
    d = str(tmp_path / ".cash")
    return Cash(cache_dir=d, register_magic=False), Cash(cache_dir=d, register_magic=False)


def model(x):
    runs.append(x)
    return x


def other(x):
    runs.append(-x)
    return -x


runs: list[int] = []


def test_a_clear_elsewhere_stops_a_running_process_serving_it(tmp_path):
    server, admin = _pair(tmp_path)
    served = server.cache(assume_safe=True)(model)
    cleared = admin.cache(assume_safe=True)(model)
    runs.clear()

    served(1)
    served(1)
    assert runs == [1]

    cleared.cache_clear()

    served(1)
    assert runs == [1, 1], "the running process kept serving what was cleared"


def test_the_clearing_process_keeps_its_other_results_in_ram(tmp_path, monkeypatch):
    c, _ = _pair(tmp_path)
    f = c.cache(assume_safe=True)(model)
    g = c.cache(assume_safe=True)(other)
    f(1)
    g(1)
    ram = c.backend.backends[0]
    drops = []
    monkeypatch.setattr(ram, "clear", lambda: drops.append(1))

    f.cache_clear()
    g(1)
    assert drops == [], "its own clear emptied its whole RAM tier"


def test_an_entry_that_cannot_be_removed_is_reported(tmp_path, monkeypatch):
    c, _ = _pair(tmp_path)
    f = c.cache(assume_safe=True)(model)
    f(1)
    c.backend.backends[1]._writes.wait_all()
    real_remove = os.remove

    def held_open(path, *a, **k):
        if path.endswith(".entry"):
            raise PermissionError(13, "The process cannot access the file", path)
        real_remove(path, *a, **k)

    monkeypatch.setattr(file_eviction, "REPLACE_RETRY_DELAYS", (0.0,))
    monkeypatch.setattr(file_eviction.os, "remove", held_open)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f.cache_clear()
    codes = [getattr(w.message, "code", None) for w in caught if issubclass(w.category, CashCacheStoreFailedWarning)]
    assert "CACHE-CLEAR-INCOMPLETE" in codes, [str(w.message) for w in caught]


def test_a_briefly_held_entry_is_removed_after_a_retry(tmp_path, monkeypatch):
    c, _ = _pair(tmp_path)
    f = c.cache(assume_safe=True)(model)
    runs.clear()
    f(1)
    c.backend.backends[1]._writes.wait_all()
    real_remove = os.remove
    refusals = [1, 1]

    def briefly_held(path, *a, **k):
        if path.endswith(".entry") and refusals:
            refusals.pop()
            raise PermissionError(13, "The process cannot access the file", path)
        real_remove(path, *a, **k)

    monkeypatch.setattr(file_eviction.os, "remove", briefly_held)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        f.cache_clear()
    assert not [w for w in caught if "CACHE-CLEAR-INCOMPLETE" in str(w.message)]
    f(1)
    assert runs == [1, 1]
