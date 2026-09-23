"""``Cash(backends=[ram, disk])`` must behave like the stack ``Cash()`` builds.

A list of backends used to be wrapped in a separate composite with no TTL
stamping, size caps, cost gate or clear detection, so the same two tiers
behaved differently depending on how they were handed to ``Cash``.
"""

from __future__ import annotations

import shutil

from cash.backends import FileBackend, InMemoryBackend, TieredBackend
from cash.backends.clear_watch import ClearWatcher
from cash.core import Cash


def test_a_backends_list_builds_the_tiered_stack(tmp_path, monkeypatch):
    cache_dir = tmp_path / "cache"
    ram = InMemoryBackend()
    disk = FileBackend(str(cache_dir))
    c = Cash(backends=[ram, disk], register_magic=False)
    assert isinstance(c.backend, TieredBackend)
    assert c.backend.backends == [ram, disk]
    monkeypatch.setattr(ClearWatcher, "CHECK_EVERY", 0.0)

    calls = []

    @c.cache
    def double(x):
        calls.append(x)
        return x * 2

    assert double(21) == 42
    assert double(21) == 42
    assert calls == [21]

    # `cash clear --all` removes the directory under a running process. The
    # RAM tier must notice and stop serving the cleared result.
    disk._writes.wait_all()
    c._stored_keys.flush()  # and the stored-key record's
    shutil.rmtree(cache_dir)
    assert double(21) == 42
    assert calls == [21, 21], "the RAM tier served a result cleared from disk"
