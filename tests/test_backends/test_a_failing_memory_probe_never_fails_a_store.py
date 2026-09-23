"""The RAM tier's memory-pressure check never fails the write it runs after.

The check reads the machine's memory on every ``check_interval``-th write,
after the value is already stored. It let through anything but ``OSError``
and ``AttributeError``, so a psutil left patched by an earlier test -- its
stand-in read a name that no longer existed -- made every tenth write in the
kernel raise ``NameError``. A reading that fails now skips the check, and says
so once.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

from cash.backends import memory_backend
from cash.backends.memory_backend import InMemoryBackend


def _broken(exc: BaseException):
    def virtual_memory():
        raise exc

    return SimpleNamespace(virtual_memory=virtual_memory)


@pytest.fixture(autouse=True)
def _not_yet_logged(monkeypatch):
    monkeypatch.setattr(memory_backend, "_READING_FAILURE_LOGGED", [], raising=False)


@pytest.mark.parametrize("exc", [NameError("name '_ty' is not defined"), RuntimeError("no /proc"), OSError("denied")])
def test_every_write_stores_when_the_reading_raises(monkeypatch, caplog, exc):
    monkeypatch.setattr(memory_backend, "psutil", _broken(exc))
    b = InMemoryBackend(check_interval=10)

    with caplog.at_level(logging.WARNING, logger=memory_backend.__name__):
        for i in range(25):
            b.set(f"k{i}", i, {"execution_time": 1.0})

    assert all(b.get(f"k{i}")[1] == i for i in range(25))
    logged = [r for r in caplog.records if "memory-pressure check" in r.getMessage()]
    assert len(logged) == 1, "said once, not once per check: %r" % [r.getMessage() for r in logged]
    assert type(exc).__name__ in logged[0].getMessage()


def test_a_reading_without_a_percentage_skips_the_check(monkeypatch):
    monkeypatch.setattr(memory_backend, "psutil", SimpleNamespace(virtual_memory=lambda: SimpleNamespace(percent=None)))
    b = InMemoryBackend(check_interval=1)

    for i in range(5):
        b.set(f"k{i}", i, {"execution_time": 1.0})

    assert b.entry_count() == 5


def test_a_working_reading_still_evicts(monkeypatch):
    """Control: the check itself still runs when the reading works."""
    reading = SimpleNamespace(percent=50.0, total=16_000_000_000)
    monkeypatch.setattr(memory_backend, "psutil", SimpleNamespace(virtual_memory=lambda: reading))
    b = InMemoryBackend(max_memory_percent=0.9, check_interval=1)
    for i in range(100):
        b.set(f"k{i:03d}", bytes(1_000_000), {"execution_time": 1.0})

    reading.percent = 95.0
    b._check_and_evict()

    assert b.entry_count() < 100
