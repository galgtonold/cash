"""Three more readers record what they read.

Found while attacking the decorator before round 26, each returning a stale
answer with no warning after the data changed:

* ``pyarrow.dataset.dataset(dir)`` -- while ``pyarrow.parquet.read_table`` on
  the same directory was correct,
* ``linecache``, which reads through a reference captured at import time and so
  bypasses the patched ``open``. Source files stay out of it: linecache is what
  ``inspect.getsource`` and every traceback read with.

``os.open`` -- the descriptor-level API below ``open()`` -- is the third of the
family and is deliberately NOT tracked: cash's own storage and the import
machinery read through it, and recording those made a module's ``.pyc`` and the
cache's own entries look like a function's data. It stays a documented gap.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap

import pytest

PROGRAM = textwrap.dedent('''
    import json, linecache, os, time
    import cash
    cash.configure(cache_dir=CACHE)
    D = DATA

    @cash.cache
    def via_dataset():
        time.sleep(0.3)
        import pyarrow.dataset as ds
        return int(ds.dataset(os.path.join(D, "parts"), format="parquet").to_table().num_rows)

    @cash.cache
    def via_linecache():
        time.sleep(0.3)
        path = os.path.join(D, "raw.txt")
        linecache.checkcache(path)
        return linecache.getline(path, 1).strip()

    print(json.dumps([via_dataset(), via_linecache()]))
''')


def _data(tmp_path, rows, text):
    pa = pytest.importorskip("pyarrow")
    import pyarrow.parquet as pq
    parts = tmp_path / "data" / "parts"
    parts.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.table({"x": list(range(rows))}), str(parts / f"p{rows}.parquet"))
    (tmp_path / "data" / "raw.txt").write_text(text + "\n", encoding="utf-8")


def _run(tmp_path):
    script = tmp_path / "run.py"
    script.write_text(
        PROGRAM.replace("CACHE", repr(str(tmp_path / ".cash"))).replace("DATA", repr(str(tmp_path / "data"))),
        encoding="utf-8")
    done = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                          timeout=180, cwd=str(tmp_path))
    assert done.returncode == 0, done.stderr[-1500:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.timeout(300)
def test_each_reader_sees_its_data_change(tmp_path):
    _data(tmp_path, 3, "AAAAA")
    assert _run(tmp_path) == [3, "AAAAA"]
    _data(tmp_path, 100, "BBBBB")
    assert _run(tmp_path) == [103, "BBBBB"]


def test_a_source_file_read_through_linecache_is_not_a_dependency(tmp_path):
    """The guard for the above: reading a .py through linecache -- which is what
    inspect.getsource does -- must not make that module a data dependency."""
    import linecache

    from cash.notebook.file_tracker import FileAccessTracker

    module = tmp_path / "some_module.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    data = tmp_path / "rows.txt"
    data.write_text("one\n", encoding="utf-8")

    tracker = FileAccessTracker(user_ns={})
    with tracker:
        linecache.checkcache(str(module))
        linecache.getline(str(module), 1)
        linecache.checkcache(str(data))
        linecache.getline(str(data), 1)
    tracked = {str(f).replace("\\", "/") for f in tracker.get_accessed_files()}
    assert not [t for t in tracked if t.endswith(".py")], tracked
    assert any(t.endswith("rows.txt") for t in tracked), tracked


def test_a_pseudo_filename_is_not_a_dependency(tmp_path):
    """The other guard: code compiled from memory has no file behind it.

    Every notebook statement cash runs is compiled under a name like
    ``<cash-0beec9249e1e>``, as `<string>`, `<stdin>` and `<ipython-input-3>`
    are elsewhere. Tracking those gave each statement a dependency on a file
    that cannot exist, and the planner read that as "still satisfied": an
    edited cell kept the previous run's live generator and averaged five values
    instead of two (avg=72.0 where the oracle says 150.0).
    """
    import linecache

    from cash.notebook.file_tracker import FileAccessTracker

    data = tmp_path / "rows.txt"
    data.write_text("one\n", encoding="utf-8")

    tracker = FileAccessTracker(user_ns={})
    with tracker:
        for name in ("<cash-0beec9249e1e>", "<string>", "<stdin>", "<ipython-input-3-abc>"):
            linecache.getlines(name)
            linecache.getline(name, 1)
        linecache.checkcache(str(data))
        linecache.getline(str(data), 1)
    tracked = {str(f).replace("\\", "/") for f in tracker.get_accessed_files()}
    assert not [t for t in tracked if "<" in t], tracked
    assert any(t.endswith("rows.txt") for t in tracked), tracked
