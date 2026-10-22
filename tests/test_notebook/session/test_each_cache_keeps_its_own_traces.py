"""Each notebook's cache keeps its own what-ran traces.

The shadow asked the backend for ``local_dir()``, a property on the backends:
the call raised, the error was taken for "no local folder", and every
notebook on the machine wrote its traces to one ``shadow`` folder under the
system temp dir, where the traces kept per statement were counted across
notebooks.
"""

from __future__ import annotations

import os
import tempfile

from cash.backends import FileBackend, InMemoryBackend, TieredBackend
from cash.notebook import shadow


def _backend(path):
    return TieredBackend([InMemoryBackend(), FileBackend(str(path))])


def test_two_notebooks_with_their_own_caches_keep_their_traces_apart(tmp_path):
    first, second = shadow.Shadow(), shadow.Shadow()
    first.attach({"__name__": "__main__"}, {}, _backend(tmp_path / "a"))
    second.attach({"__name__": "__main__"}, {}, _backend(tmp_path / "b"))

    assert first.dir == os.path.join(str(tmp_path / "a"), "shadow")
    assert second.dir == os.path.join(str(tmp_path / "b"), "shadow")
    assert not first.dir.startswith(tempfile.gettempdir() + os.sep + "shadow")


def test_a_cache_with_no_local_folder_keeps_its_traces_in_the_temp_dir(monkeypatch):
    monkeypatch.delenv("CASH_CACHE_DIR", raising=False)
    kept = shadow.Shadow()
    kept.attach({"__name__": "__main__"}, {}, InMemoryBackend())

    assert kept.dir == os.path.join(tempfile.gettempdir(), "shadow")
