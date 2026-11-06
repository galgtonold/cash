"""A trivial cell below a loop over many files costs what changed, not what was read.

A sentiment notebook (E10) unpacks 50,000 review files, reads each into a
frame in a loop, writes the frame to ``movie_data.csv`` and reads it back.
On Windows every cell below, ``df.shape`` alike, then cost 2-3.6 s where
plain Python takes 0.05 s: before each one the upstream check worked out
again, from every one of the 100,000 paths read, which of them the writer's
file is, each path's comparable spellings, and the file component of the
statements over the frame -- a stat or a look-up in the folder watch of
each, and their spellings, in Python.

These tests run the notebook's shape through cash with a few thousand
files and count the work of each later cell on them: the paths whose stats
are asked for, each look at a file (a stat, an ``lstat``, a look-up in the
watch), each spelling of a path (``normcase``, ``split``, ``realpath``).
A cell below costs a few of each, whatever the count of files; a file
changed among them is still seen.
"""

from __future__ import annotations

import collections
import io
import os
import sys
import tarfile

import pytest

from cash.notebook import run_memo
from cash.tracking import file_watch
from tests._cell_driver import run_cash_cell

from .test_a_held_check_of_the_cells_above_looks_at_no_file import _AsOnWindows, _listed

pytestmark = pytest.mark.skipif(sys.platform == "darwin", reason="folders are not watched on macOS")

pd = pytest.importorskip("pandas")

#: Reviews in each of the two folders: past `LISTING_MIN_FILES` many times over.
FILES = 1500

#: What a cell below may look at: the notebook's own few paths, each watched folder.
FEW = 48

#: ``filter=`` came in 3.10.12; before it, ``extractall`` does what ``'fully_trusted'`` does.
EXTRACT_ARGS = "filter='fully_trusted'" if hasattr(tarfile, "fully_trusted_filter") else ""

CELLS = [
    "import os, tarfile\nimport pandas as pd",
    f"if not os.path.isdir('aclImdb'):\n    with tarfile.open('data.tar') as tar:\n        tar.extractall({EXTRACT_ARGS})",
    "rows = []\n"
    "for label in ('pos', 'neg'):\n"
    "    path = os.path.join('aclImdb', 'train', label)\n"
    "    for name in sorted(os.listdir(path)):\n"
    "        with open(os.path.join(path, name), 'r', encoding='utf-8') as fh:\n"
    "            rows.append((fh.read(), label))\n"
    "df = pd.DataFrame(rows, columns=['review', 'sentiment'])",
    "df = df.sample(frac=1, random_state=0).reset_index(drop=True)",
    "df.to_csv('movie_data.csv', index=False, encoding='utf-8')",
    "df = pd.read_csv('movie_data.csv', encoding='utf-8')\ndf = df.rename(columns={'0': 'review'})\ndf.head(3)",
    "shape = df.shape",
    "n = len(df)",
    "edited = int(df['review'].str.contains('EDITED').sum())",
    "m = n + 1",
    "edited_rows = sum('EDITED' in review for review, _label in rows)",
]

TRIVIAL = CELLS[6:]


@pytest.fixture(params=["as on Linux", "as on Windows"])
def backend(request, monkeypatch):
    """The folder watch as on Linux, or as on Windows (`file_watch._Volume`
    standing in for kernel32, a crowded folder's files stat-ed by listing
    it), where the test makes the reports; None on Linux."""
    if request.param == "as on Linux":
        return None
    backend = _AsOnWindows()
    monkeypatch.setattr(file_watch, "_backend", lambda: backend)
    monkeypatch.setattr(run_memo, "stats_from_listings", _listed)
    return backend


@pytest.fixture
def notebook(cash_magics, tmp_path, monkeypatch, backend):
    """The cells above run once, and the trivial ones twice (what is held
    is taken on the first, then holds); yields a runner of one cell."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CASH_FILE_WATCH", raising=False)
    file_watch.reset()
    with tarfile.open(tmp_path / "data.tar", "w") as tar:
        for label in ("pos", "neg"):
            for i in range(FILES):
                data = f"review {label} {i}".encode()
                info = tarfile.TarInfo(f"aclImdb/train/{label}/{i}_7.txt")
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

    def run(code: str) -> None:
        run_cash_cell(cash_magics, code, cells=CELLS)

    for code in CELLS + TRIVIAL:
        run(code)
    yield run
    file_watch.reset()


@pytest.fixture
def work(monkeypatch, tmp_path) -> collections.Counter:
    """Each look at a path under the project (or relative), each path whose
    stats are asked for (`run_memo.stats_this_run`), and each spelling of
    one (``normcase``, ``split``, ``realpath``), from `start` on."""
    seen: collections.Counter = collections.Counter()
    root = str(tmp_path)

    def ours(path) -> bool:
        try:
            spelled = os.fspath(path)
        except TypeError:
            return False
        return isinstance(spelled, str) and (spelled.startswith(root) or not os.path.isabs(spelled))

    def counting(kind, real):
        def call(path, *args, **kwargs):
            if ours(path):
                seen[kind] += 1
            return real(path, *args, **kwargs)

        return call

    def counting_stats(real):
        def call(paths, *args, **kwargs):
            if not isinstance(paths, (dict, set, frozenset, list, tuple)):
                paths = list(paths)
            seen["stats asked"] += len(paths)
            return real(paths, *args, **kwargs)

        return call

    def start():
        for kind in ("stat", "lstat"):
            monkeypatch.setattr(os, kind, counting(kind, getattr(os, kind)))
        for kind in ("normcase", "split", "realpath"):
            monkeypatch.setattr(os.path, kind, counting(kind, getattr(os.path, kind)))
        watch = file_watch.watcher()
        monkeypatch.setattr(watch, "current", counting("current", watch.current))
        for module in _users_of_stats_this_run():
            monkeypatch.setattr(module, "stats_this_run", counting_stats(run_memo.stats_this_run))

    seen.start = start  # type: ignore[attr-defined]
    return seen


def _users_of_stats_this_run() -> list:
    from cash.notebook.upstream import file_writers, statement_lineage

    return [run_memo, file_writers, statement_lineage]


def test_each_trivial_cell_below_looks_at_a_few_paths(notebook, work, backend, tmp_path):
    assert file_watch.watcher() is not None, "no folder watch: the count below would prove nothing"
    work.start()

    for code in TRIVIAL:
        if backend is not None:
            # What the listing (or anyone's) of each folder brings on.
            backend.report(*((3, tmp_path / "aclImdb" / "train" / label) for label in ("pos", "neg")))
        before = dict(work)
        notebook(code)
        cell = {k: v - before.get(k, 0) for k, v in work.items()}
        # Each kind of work a few times, not once per review file (3,000
        # files, read by two spellings each).
        assert all(v <= FEW for v in cell.values()), (code, cell)


def test_a_review_edited_among_them_is_seen(notebook, tmp_path, cash_magics, backend):
    review = tmp_path / "aclImdb" / "train" / "neg" / "17_7.txt"
    review.write_text("EDITED review", encoding="utf-8")
    if backend is not None:
        backend.report((3, review))

    notebook("edited_rows = sum('EDITED' in review for review, _label in rows)")

    # As a restart and Run All leave it: the loop reads the edit.
    assert cash_magics.shell.user_ns["edited_rows"] == 1
