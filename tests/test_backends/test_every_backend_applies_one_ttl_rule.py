"""Every place that decides whether an entry has expired applies one rule.

A ttl the decorator declared stands as written; any other is the shorter of
the ttl the entry was written with and the tier's ``default_ttl``, counted
from when the entry was written. A bare file or SQLite backend served
entries that ``cash clear --expired`` and a tiered read call expired, and
the two disagreed with each other.
"""

from __future__ import annotations

import time

import pytest

from cash.backends._base import entry_expired
from cash.backends.file_backend import FileBackend
from cash.backends.sqlite_backend import SQLiteBackend


@pytest.fixture(params=["file", "sqlite"])
def backend(request, tmp_path):
    if request.param == "file":
        b = FileBackend(str(tmp_path / "c"), default_ttl=10, flush_interval=0)
    else:
        b = SQLiteBackend(db_path=str(tmp_path / "c.db"), default_ttl=10)
    yield b
    b.shutdown()


CASES = [
    # (written metadata, expired under a tier default of 10 s at an age of 100 s)
    ({"ttl": None}, True),
    ({"ttl": 1000}, True),
    ({"ttl": 1000, "ttl_declared": True}, False),
    ({"ttl": 5}, True),
]


def test_a_bare_backend_expires_what_the_shared_rule_expires(backend):
    backend.set("fresh", 1, {"ttl": None})
    assert backend.get("fresh")[1] == 1, "control: a fresh entry is served"

    for i, (written, expired) in enumerate(CASES):
        metadata = dict(written, created_at=time.time() - 100)
        assert entry_expired(metadata, 10) is expired
        backend.set(f"k{i}", 1, dict(metadata))
        assert (backend.get_metadata(f"k{i}") is None) is expired, written
        assert (backend.get(f"k{i}")[0] is None) is expired, written
