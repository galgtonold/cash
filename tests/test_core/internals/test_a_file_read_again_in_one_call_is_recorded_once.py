"""A file a cached call reads again is recorded once, and recorded the same.

Recording a read resolves the path, classifies it by the frames around it
and credits it to the user code on the stack. A cached call that opened one
file 50,000 times paid that 50,000 times: ~60 us an ``open`` where plain
Python takes 8 us. A read the block has recorded already -- the same path,
from the same working directory, by the same code, of a file whose stat is
what the block first saw -- has nothing left to record, and is answered by
one ``lstat``.

What the entry depends on must not move: the same files, stats and digests
as when every read is recorded in full.
"""

from __future__ import annotations

import functools
import os
import warnings

import pytest

from cash import Cash
from cash.tracking import read_credit
from cash.tracking.file_tracker import FileAccessTracker
from tests._work_counts import method_calls

pytestmark = [pytest.mark.core]


def _write(path, text):
    path.write_text(text, encoding="utf-8")
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns - 60_000_000_000))  # settled: not just written


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def _read_many(paths, rounds):
    total = 0
    for _ in range(rounds):
        for path in paths:
            total += len(_read(path))
    return total


def test_a_file_opened_again_and_again_is_recorded_in_full_once(tmp_path):
    data = tmp_path / "data.txt"
    _write(data, "x" * 100)
    c = Cash(cache_dir=str(tmp_path / "cache"))
    cached = c.cache(_read_many)
    with method_calls(FileAccessTracker, "_track_path_untimed") as full:
        assert cached((str(data),), 1000) == 100_000
    # The first read, and a second while the stack's anchor settles.
    assert 1 <= full.calls <= 3, full.calls
    with method_calls(FileAccessTracker, "_track_path_untimed") as hit:
        assert cached((str(data),), 1000) == 100_000  # served: the entry names the file
    assert hit.calls == 0


def test_many_files_read_in_rounds_are_each_recorded_in_full_once(tmp_path):
    paths = []
    for i in range(40):
        path = tmp_path / f"f{i}.txt"
        _write(path, "x" * 10)
        paths.append(str(path))
    c = Cash(cache_dir=str(tmp_path / "cache"))
    cached = c.cache(_read_many)
    with method_calls(FileAccessTracker, "_track_path_untimed") as full:
        assert cached(tuple(paths), 25) == 40 * 25 * 10
    assert 40 <= full.calls <= 40 + 3, full.calls


def _snapshot(tracker):
    return (
        sorted(tracker.get_accessed_files()),
        # By size: the two runs read two copies of the files, written apart.
        {path: st[0] for path, st in tracker.read_stats.items()},
        dict(tracker.read_digests),
        sorted(tracker.get_absent_files()),
        sorted(tracker.get_unresolved_files()),
    )


def _scenario_repeat(tmp_path):
    for _ in range(5):
        _read(str(tmp_path / "a.txt"))
        _read("b.txt")


def _scenario_changed_between(tmp_path):
    _read(str(tmp_path / "a.txt"))
    _read(str(tmp_path / "a.txt"))
    (tmp_path / "a.txt").write_text("changed while the block ran", encoding="utf-8")
    _read(str(tmp_path / "a.txt"))


def _scenario_gone_between(tmp_path):
    _read(str(tmp_path / "a.txt"))
    _read(str(tmp_path / "a.txt"))
    os.unlink(tmp_path / "a.txt")
    try:
        _read(str(tmp_path / "a.txt"))
    except FileNotFoundError:
        pass


def _scenario_chdir(tmp_path):
    _read("b.txt")
    _read("b.txt")
    os.chdir(tmp_path / "sub")
    _read("b.txt")  # another file by the same name
    _read("b.txt")


def _scenario_link_repointed(tmp_path):
    link = tmp_path / "link.txt"
    try:
        os.symlink(tmp_path / "a.txt", link)
    except (OSError, NotImplementedError):
        pytest.skip("no symlinks here")
    _read(str(link))
    _read(str(link))
    os.unlink(link)
    os.symlink(tmp_path / "b.txt", link)
    _read(str(link))


@pytest.mark.parametrize(
    "scenario",
    [_scenario_repeat, _scenario_changed_between, _scenario_gone_between, _scenario_chdir, _scenario_link_repointed],
)
def test_the_block_records_what_it_records_with_every_read_in_full(tmp_path, monkeypatch, scenario):
    """Each shape run twice over the same files: once as shipped, once with
    the short cut answering "not recorded" every time."""
    snapshots = []
    for short_cut in (True, False):
        root = tmp_path / ("fast" if short_cut else "full")
        (root / "sub").mkdir(parents=True)
        _write(root / "a.txt", "aaa")
        _write(root / "b.txt", "bbbb")
        _write(root / "sub" / "b.txt", "another b")
        with pytest.MonkeyPatch.context() as mp:
            mp.chdir(root)
            if not short_cut:
                mp.setattr(FileAccessTracker, "recorded_unchanged", lambda self, read: False, raising=False)
            tracker = FileAccessTracker(hash_on_read=True)
            with method_calls(FileAccessTracker, "_track_path_untimed") as full:
                with tracker:
                    scenario(root)
            snapshot = _snapshot(tracker)
        snapshots.append((str(snapshot).replace(str(root), "<root>"), full.calls))
    (fast, fast_calls), (full_, full_calls) = snapshots
    assert fast == full_
    assert "a.txt" in fast or "b.txt" in fast  # positive control: something was recorded
    if scenario is _scenario_repeat:
        assert fast_calls < full_calls == 10  # and the short cut was taken


def test_a_second_reader_of_a_recorded_file_is_still_credited(tmp_path):
    """The key names the code that reads: another function reading the same
    file in the same block is credited the read like the first."""
    data = tmp_path / "data.txt"
    _write(data, "abc")

    def first(path):
        return _read(path)

    def second(path):
        return _read(path)

    with FileAccessTracker():
        for _ in range(3):
            first(str(data))
        for _ in range(3):
            second(str(data))
    resolved = os.path.realpath(data).replace("\\", "/")
    assert resolved in {p.replace("\\", "/") for p in read_credit.credited_reads(first.__code__)}
    assert resolved in {p.replace("\\", "/") for p in read_credit.credited_reads(second.__code__)}


def test_a_file_that_changes_between_two_reads_of_one_call_is_not_stored(tmp_path):
    data = tmp_path / "data.txt"
    _write(data, "old")
    c = Cash(cache_dir=str(tmp_path / "cache"))
    runs = []

    @c.cache
    def content(path):
        runs.append(1)
        first = _read(path)
        _read(path)
        # Another writer, as far as the call can tell: not through open().
        fd = os.open(path, os.O_WRONLY | os.O_TRUNC)  # @cash:assume-safe
        os.write(fd, b"new text")
        os.close(fd)
        return first + _read(path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        assert content(str(data)) == "oldnew text"
        _write(data, "old")
        assert content(str(data)) == "oldnew text"
    assert len(runs) == 2, "a result computed from two versions of its input was stored"
    assert "STORE-INPUT-CHANGED" in [getattr(w.message, "code", None) for w in caught]


def test_a_memoised_helper_read_again_still_invalidates(tmp_path):
    data = tmp_path / "data.txt"
    _write(data, "old")
    load = functools.lru_cache(maxsize=None)(_read)
    c = Cash(cache_dir=str(tmp_path / "cache"))

    @c.cache
    def twice(path):
        return _read(path) + _read(path) + load(path)

    assert twice(str(data)) == "oldoldold"
    _write(data, "new")
    load.cache_clear()
    assert twice(str(data)) == "newnewnew"
