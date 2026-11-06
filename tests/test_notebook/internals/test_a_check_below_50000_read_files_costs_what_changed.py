"""A check below a cell that read 50,000 files costs what changed, not what was read.

A sentiment notebook reads 50,000 review files in a loop, then round-trips
the frame through one CSV beside the notebook. On Windows every cell below
it, ``df.shape`` and ``porter = PorterStemmer()`` alike, then cost 1-5 s of
checking those files. These tests count the work of one cell's checks with
the volume watch as on Windows (`file_watch._Volume`, standing in for
kernel32) and the listing stats Windows takes of a crowded folder:

* the held verdict on a cached cell's 50,000 recorded keys, asked again
  with the very dict, reads none of its entries (it compared them all with
  the held copy, and looked for a None among them, twice a cell);
* a folder Windows reports modified in place, as it does after the
  listing a check took of it, has none of its files looked at again (each
  listing brought on the next report, and every cell checked all of them).
"""

from __future__ import annotations

import collections
import os
import sys

import pytest

from cash.notebook import run_memo
from cash.tracking import file_dep_snapshot, file_watch
from cash.tracking.read_classification import stat_key

from .test_a_held_check_of_the_cells_above_looks_at_no_file import _AsOnWindows, _listed

pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="folders are not watched on macOS")

#: Folders of reviews, and reviews in each: 50,000 files.
FOLDERS, FILES = 4, 12_500

#: Each check may stat each watched folder, and the CSV's folder holds a few files.
FEW = 16


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """``aclImdb/set<d>/<i>.txt``, and ``movie_data.csv`` beside the notebook."""
    root = tmp_path_factory.mktemp("sentiment")
    for d in range(FOLDERS):
        folder = root / "aclImdb" / f"set{d}"
        folder.mkdir(parents=True)
        for i in range(FILES):
            with open(folder / f"{i}_7.txt", "w", encoding="utf-8") as fh:
                fh.write("a review")
    (root / "movie_data.csv").write_text("review,sentiment\n", encoding="utf-8")
    return root


@pytest.fixture(autouse=True)
def _runs(monkeypatch, project):
    monkeypatch.chdir(project)
    monkeypatch.delenv("CASH_FILE_WATCH", raising=False)
    monkeypatch.setattr(file_dep_snapshot, "HASH_EPOCH", None)
    monkeypatch.setattr(file_dep_snapshot, "_EPOCH_DEPTH", 0)
    monkeypatch.setattr(run_memo, "stats_from_listings", _listed)
    backend = _AsOnWindows()
    monkeypatch.setattr(file_watch, "_backend", lambda: backend)
    run_memo._FILE_STATE_THIS_RUN.clear()
    run_memo._HELD_KEYS.clear()
    run_memo._COVERED_COMPONENTS.clear()
    file_watch.reset()
    file_dep_snapshot.begin_file_state_epoch()
    yield backend
    file_dep_snapshot.end_file_state_epoch()
    run_memo._FILE_STATE_THIS_RUN.clear()
    run_memo._HELD_KEYS.clear()
    run_memo._COVERED_COMPONENTS.clear()
    file_watch.reset()
    (project / "notes.ipynb").unlink(missing_ok=True)


def _next_run() -> None:
    file_dep_snapshot.end_file_state_epoch()
    file_dep_snapshot.begin_file_state_epoch()


def _reviews(project) -> list[str]:
    """Each review by both spellings, as a read by a relative path records it."""
    paths = []
    for d in range(FOLDERS):
        for i in range(FILES):
            relative = f"aclImdb/set{d}/{i}_7.txt"
            paths += [relative, str(project / relative)]
    return paths


class _Counted(dict):
    """A recorded ``{path: stat_key}`` that counts each pass over its entries."""

    passes = 0

    def _count(self) -> None:
        type(self).passes += 1

    def values(self):
        self._count()
        return super().values()

    def items(self):
        self._count()
        return super().items()

    def keys(self):
        self._count()
        return super().keys()

    def __iter__(self):
        self._count()
        return super().__iter__()

    def __eq__(self, other):
        self._count()
        return super().__eq__(other)

    __hash__ = None  # type: ignore[assignment]


@pytest.fixture
def looks(monkeypatch, project) -> collections.Counter:
    """Each stat and lstat of a path under the project, and each path the
    watch is asked about (``current``)."""
    seen: collections.Counter = collections.Counter()
    root = str(project)

    def counting(kind, real):
        def call(path, *args, **kwargs):
            spelled = os.fspath(path)
            if spelled.startswith(root) or not os.path.isabs(spelled):
                seen[kind] += 1
            return real(path, *args, **kwargs)

        return call

    def start():
        monkeypatch.setattr(os, "stat", counting("stat", os.stat))
        monkeypatch.setattr(os, "lstat", counting("lstat", os.lstat))
        watch = file_watch.watcher()
        monkeypatch.setattr(watch, "current", counting("current", watch.current))

    seen.start = start  # type: ignore[attr-defined]
    return seen


def test_a_held_verdict_asked_again_reads_none_of_the_recorded_keys(project, looks):
    recorded = _Counted({p: stat_key(os.stat(p)) for p in _reviews(project)})
    for _ in range(2):
        assert run_memo.stat_keys_as_recorded(recorded)
        _next_run()
    assert run_memo._HELD_KEYS, "no verdict was held: the count below would prove nothing"
    looks.start()
    _Counted.passes = 0

    for _ in range(3):
        assert run_memo.stat_keys_as_recorded(recorded)
        _next_run()

    assert _Counted.passes == 0, "the held verdict went through the recorded keys"
    assert looks["stat"] <= 3 * FEW and looks["lstat"] == 0 and looks["current"] == 0, looks


def test_folders_reported_modified_after_their_listing_have_no_file_looked_at(project, looks, _runs):
    backend = _runs
    recorded = {p: stat_key(os.stat(p)) for p in _reviews(project)}
    for _ in range(2):
        assert run_memo.stat_keys_as_recorded(recorded)
        _next_run()
    looks.start()

    for _ in range(3):
        # What the listing (or anyone's) of each folder brought on.
        backend.report(*((3, project / "aclImdb" / f"set{d}") for d in range(FOLDERS)))
        assert run_memo.stat_keys_as_recorded(recorded)
        _next_run()

    assert looks["stat"] + looks["lstat"] + looks["current"] <= 3 * FEW, looks
