"""The cells below a read of files the watch cannot vouch for do not redo its checks.

A dataset cache beside the project leaves each photo with a second hard
link, so the folder watch keeps no stat of it and the upstream check before
each cell looks at every file again: one stat each, which a change made
through the other link needs. On a first run (E5, 13,000 LFW photos) each
cell paid, per simulated statement below the read, besides those stats:

* the file component of every file its outputs inherited, built and then
  thrown away for the component of the files it read itself
  (`StatementLineage._apply_writes`): 2 s of a cell on Windows;
* a look at every path, per cached cell above, to hold a verdict on them
  (`run_memo._note_held_keys`, `_Held.unsure`, `_note_covered`) that the
  watch then refused, or that the next cell's check went over again.

Each is now done once per run, or once per refusal; the keys and values
still follow an edit or a deletion of a photo.
"""

from __future__ import annotations

import os
import sys

import pytest

from cash.notebook import run_memo
from cash.notebook.statement import file_deps
from cash.tracking import file_watch
from tests._cell_driver import run_cash_cell

pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="folders are not watched on macOS")

#: Past what is held at all (`statement_lineage._HOLD_MIN_FILES`).
FILES = 300

CELLS = [
    "import os\nnames = sorted(os.listdir('photos'))",
    "data = [open('photos/' + n).read() for n in names]",
    "rows = [d.upper() for d in data]",
    "a = rows[:200]",
    "b = [r + '!' for r in a]",
    "c = b[::2]",
    "d = [r.lower() for r in c]",
]


@pytest.fixture(autouse=True)
def _own_watch(monkeypatch):
    monkeypatch.delenv("CASH_FILE_WATCH", raising=False)
    file_watch.reset()
    run_memo._COVERED_SNAPSHOTS.clear()
    run_memo._COVERED_COMPONENTS.clear()
    run_memo._HELD_KEYS.clear()
    run_memo._REFUSED_KEYS.clear()
    yield
    file_watch.reset()
    run_memo._COVERED_SNAPSHOTS.clear()
    run_memo._COVERED_COMPONENTS.clear()
    run_memo._HELD_KEYS.clear()
    run_memo._REFUSED_KEYS.clear()


@pytest.fixture
def photos(tmp_path, monkeypatch):
    """Photos under a relative path, each with a second link in a cache beside it."""
    monkeypatch.chdir(tmp_path)
    folder, cache = tmp_path / "photos", tmp_path / "cache"
    folder.mkdir()
    cache.mkdir()
    for i in range(FILES):
        original = cache / f"p{i:03d}.pgm"
        original.write_text(f"photo {i:03d}", encoding="utf-8")
        try:
            os.link(original, folder / f"p{i:03d}.pgm")
        except (OSError, NotImplementedError):
            pytest.skip("no hard links here")
    return folder


@pytest.fixture
def work(monkeypatch) -> dict[str, int]:
    """How often each repeated piece of work ran, over every file."""
    counts = {"component": 0, "held_keys": 0, "covered": 0}
    real_component = file_deps.compute_file_hash_component
    real_parts = run_memo._held_keys_parts
    real_held_of = run_memo._held_of

    def component(files, *args, **kwargs):
        if len(files) >= FILES:
            counts["component"] += 1
        return real_component(files, *args, **kwargs)

    def parts(watch, recorded, *args, **kwargs):
        if len(recorded) >= FILES:
            counts["held_keys"] += 1
        return real_parts(watch, recorded, *args, **kwargs)

    def held_of(watch, snap, *args, **kwargs):
        if len(snap) >= FILES:
            counts["covered"] += 1
        return real_held_of(watch, snap, *args, **kwargs)

    monkeypatch.setattr(file_deps, "compute_file_hash_component", component)
    monkeypatch.setattr(run_memo, "_held_keys_parts", parts)
    monkeypatch.setattr(run_memo, "_held_of", held_of)
    return counts


def _run(magics, cell: str) -> None:
    run_cash_cell(magics, cell, cells=CELLS)


def _plain() -> dict:
    """What restart and Run All gives in plain Python."""
    ns: dict = {}
    for cell in CELLS:
        exec(cell, ns)
    return {name: ns[name] for name in ("rows", "a", "b", "c", "d")}


def _ns(magics) -> dict:
    return {name: magics.shell.user_ns[name] for name in ("rows", "a", "b", "c", "d")}


def test_the_cells_below_redo_no_check_per_statement(cash_magics, photos, work):
    assert file_watch.watcher() is not None, "no folder watch: the counts below would prove nothing"
    for cell in CELLS[:3]:
        _run(cash_magics, cell)
    for key in work:
        work[key] = 0

    below = CELLS[3:]
    for cell in below:
        _run(cash_magics, cell)

    assert _ns(cash_magics) == _plain()
    # Each statement below the read that runs in the simulation built the
    # component of all the photos it inherited, and threw it away.
    assert work["component"] == 0, work
    # The verdicts on the cached cells' photos: looked over at most once a
    # cell, where each cached cell above, and each statement, looked again.
    assert work["held_keys"] <= len(below), work
    assert work["covered"] <= len(below), work


def test_an_edit_or_a_deletion_still_changes_the_cells_below(cash_magics, photos):
    for cell in CELLS:
        _run(cash_magics, cell)
    lineages = [cash_magics.tracking_state.lineage["d"]]
    assert _ns(cash_magics) == _plain()

    target = photos / "p123.pgm"
    st = os.stat(target)
    with open(target, "a", encoding="utf-8") as fh:
        fh.write(" EDITED")
    os.utime(target, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    _run(cash_magics, CELLS[-1])
    assert _ns(cash_magics) == _plain()
    lineages.append(cash_magics.tracking_state.lineage["d"])

    os.unlink(photos / "p007.pgm")
    _run(cash_magics, CELLS[-1])
    assert _ns(cash_magics) == _plain()
    lineages.append(cash_magics.tracking_state.lineage["d"])

    assert len(set(lineages)) == 3, "a change to the photos left the key of a cell below as it was"
