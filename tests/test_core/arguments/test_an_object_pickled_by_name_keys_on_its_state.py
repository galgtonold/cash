"""An object that pickles by name is keyed by the state it holds.

A logger pickles as ``getLogger(name)``, which loads whatever the name finds
then: its level, handlers and parent are not in the pickle. Keyed by its
name alone, ``level_of({'log': log})`` was served 30 after
``log.setLevel(10)``. Such an object is keyed with its attributes, as a
pickle by value would hold them; a cache it fills as it is read is not one
of them, and functions, classes and bound methods (pickled by name, with no
attributes of their own to key) key as before.
"""

from __future__ import annotations

import logging

import pytest

from cash import Cash
from cash.backends import InMemoryBackend
from cash.stored_pickle import object_digest
from cash.value_hash import compute_hash


@pytest.fixture
def cash():
    return Cash(backend=InMemoryBackend(), register_magic=False)


@pytest.fixture
def log(monkeypatch):
    logger = logging.getLogger("cash_test_pickled_by_name")
    monkeypatch.setattr(logger, "level", 30)
    monkeypatch.setattr(logging.root, "handlers", [])
    return logger


def test_a_cached_call_reads_the_loggers_new_level(cash, log):
    runs = []

    @cash.cache
    def level_of(cfg):
        runs.append(1)
        return cfg["a"]["log"].level

    cfg = {"a": {"log": log}}
    assert level_of(cfg) == 30
    log.setLevel(10)
    assert level_of(cfg) == 10
    assert len(runs) == 2


@pytest.mark.parametrize("digest", [compute_hash, object_digest], ids=["key", "store"])
def test_the_digest_moves_with_the_level_and_not_with_a_log_call(log, digest):
    cfg = {"a": [log]}
    before = digest(cfg)
    log.warning("filled into the logger's cache of enabled levels")
    assert digest(cfg) == before
    log.setLevel(10)
    assert digest(cfg) != before


class Singleton:
    """Pickled by its name, as a module-level singleton is."""

    def __reduce__(self):
        return "SINGLETON"


SINGLETON = Singleton()


def test_a_users_object_pickled_by_its_name_keys_on_its_attributes(monkeypatch):
    monkeypatch.setattr(SINGLETON, "setting", 1, raising=False)
    before = compute_hash({"s": SINGLETON})
    SINGLETON.setting = 2
    assert compute_hash({"s": SINGLETON}) != before
