"""A cached call reading an object that pickles by name is keyed by its state.

A logger pickles as ``getLogger(name)``, which loads whatever the name finds
then: its level, handlers and parent are not in the pickle. Keyed by its
name alone, ``level_of({'log': log})`` was served 30 after
``log.setLevel(10)``, and so was a function reading the logger as a global.
An argument, or a global a cached function reads, holding such an object is
keyed with its attributes, as a pickle by value would hold them.
"""

from __future__ import annotations

import logging

import pytest

from cash import Cash
from cash.backends import InMemoryBackend
from cash.value_hash import compute_hash

NAME = "cash_test_pickled_by_name"


@pytest.fixture
def cash():
    return Cash(backend=InMemoryBackend(), register_magic=False)


@pytest.fixture
def log(monkeypatch):
    logger = logging.getLogger(NAME)
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
    assert level_of(cfg) == 30
    logging.getLogger(NAME).setLevel(10)
    assert level_of(cfg) == 10
    assert len(runs) == 2


LOG = logging.getLogger(NAME)
GLOBAL_RUNS: list[int] = []


def global_level():
    GLOBAL_RUNS.append(1)
    return LOG.level


def test_a_cached_function_reading_the_logger_as_a_global(cash, log):
    GLOBAL_RUNS.clear()
    level = cash.cache(global_level)
    assert level() == 30
    logging.getLogger(NAME).setLevel(10)
    assert level() == 10
    assert len(GLOBAL_RUNS) == 2


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


class Registry:
    """A library's object found again by a lookup, as a logger is, whose
    state is what it holds in a private attribute named as a cache."""

    def __init__(self) -> None:
        self._cache: dict[str, int] = {}

    def __reduce__(self):
        return find_registry, ("main",)


Registry.__module__ = "logging"
REGISTRY = Registry()


def find_registry(name: str) -> Registry:
    return REGISTRY


def test_what_a_library_object_holds_as_a_cache_is_keyed(cash, monkeypatch):
    """A private attribute named as a cache is keyed as any other: its name
    says nothing of whether it is derived from the rest."""
    monkeypatch.setattr(REGISTRY, "_cache", {"rows": 1})
    runs = []

    @cash.cache
    def rows(reg):
        runs.append(1)
        return reg._cache["rows"]

    assert rows(REGISTRY) == 1
    REGISTRY._cache["rows"] = 2
    assert rows(REGISTRY) == 2
    assert len(runs) == 2
