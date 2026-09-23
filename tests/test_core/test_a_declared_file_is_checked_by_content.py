"""``file_depends_on=`` checks the file's content, like a tracked read.

A declared file used to become a ``FileDataSource`` folded into the key, and
that keyed on the mtime alone: a ``touch`` with identical content recomputed,
and an edit that left the mtime where it was served the old result. A file the
function reads itself was already checked by content, so the two ways of
naming the same dependency disagreed.
"""

from __future__ import annotations

import os

from cash import Cash
from cash.backends.memory_backend import InMemoryBackend


def _make(tmp_path):
    c = Cash(backend=InMemoryBackend(), register_magic=False)
    path = tmp_path / "config.txt"
    path.write_text("alpha")
    runs = []

    @c.cache(file_depends_on=str(path))
    def load():
        # The body never opens the file: that is what file_depends_on is for.
        runs.append(1)
        return len(runs)

    return path, load, runs


def test_an_edit_that_keeps_the_mtime_recomputes(tmp_path):
    path, load, runs = _make(tmp_path)
    load()
    st = os.stat(path)
    path.write_text("omega")
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns))
    load()
    assert len(runs) == 2, "an edit with the mtime restored served the old result"


def test_a_touch_with_the_same_content_hits(tmp_path):
    path, load, runs = _make(tmp_path)
    load()
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    load()
    assert len(runs) == 1, "a touch that changed no content recomputed"


def test_an_edit_recomputes(tmp_path):
    path, load, runs = _make(tmp_path)
    load()
    path.write_text("a different length")
    load()
    assert len(runs) == 2


def test_re_pointing_the_declaration_recomputes(tmp_path):
    c = Cash(backend=InMemoryBackend(), register_magic=False)
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.txt").write_text("b")
    runs = []

    def body():
        runs.append(1)
        return len(runs)

    c.cache(file_depends_on=str(tmp_path / "a.txt"))(body)()
    c.cache(file_depends_on=str(tmp_path / "b.txt"))(body)()
    assert len(runs) == 2, "an entry that recorded a.txt hit for a function now declaring b.txt"
