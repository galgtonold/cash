"""Cache-format versioning for the on-disk FileBackend.

The on-disk layout (one ``*.entry`` file per entry) can change shape
between Cash versions. To avoid silently decoding a stale layout after an
upgrade, FileBackend stamps the cache directory with the format version it
wrote and, on init, **auto-invalidates** any cache whose stamp doesn't match
the version this build expects, so nobody has to remember to clear the cache
by hand after upgrading.
"""

from __future__ import annotations

import glob
import os

from cash.backends.cache_dir import CACHE_FORMAT_VERSION
from cash.backends.entry_format import ENTRY_SUFFIX
from cash.backends.file_backend import FileBackend

VERSION_FILENAME = "CACHE_VERSION"


def _entry_files(cache_dir: str) -> list[str]:
    """Every entry file: one ``*.entry`` per entry."""
    return glob.glob(os.path.join(cache_dir, f"*{ENTRY_SUFFIX}"))


def test_fresh_cache_stamps_current_version(tmp_path):
    """A brand-new cache dir gets the current format stamp, no wipe."""
    backend = FileBackend(str(tmp_path))
    backend.set("k", {"v": 1})
    meta, value = backend.get("k")
    assert value == {"v": 1}

    version_file = tmp_path / VERSION_FILENAME
    assert version_file.exists()
    assert version_file.read_text(encoding="utf-8").strip() == str(CACHE_FORMAT_VERSION)


def test_matching_version_preserves_entries(tmp_path):
    """Reopening a cache written by the same format version keeps the data
    (the restart-and-restore path must not wipe a compatible cache)."""
    b1 = FileBackend(str(tmp_path))
    b1.set("k", {"v": 1})
    b1.get("k")  # force a write + flush of the value
    b1.shutdown()

    # Simulate a kernel restart: a new backend over the same directory.
    b2 = FileBackend(str(tmp_path))
    meta, value = b2.get("k")
    assert value == {"v": 1}, "compatible cache should survive a reopen"


def test_stale_version_marker_wipes_cache(tmp_path):
    """An older format stamp triggers auto-invalidation of existing entries."""
    backend = FileBackend(str(tmp_path))
    backend.set("k", {"v": 1})
    backend.get("k")
    backend.shutdown()
    assert _entry_files(str(tmp_path)), "precondition: entries exist on disk"

    # Pretend the cache was written by an older, incompatible format.
    (tmp_path / VERSION_FILENAME).write_text(str(CACHE_FORMAT_VERSION - 1), encoding="utf-8")

    reopened = FileBackend(str(tmp_path))
    meta, value = reopened.get("k")
    assert value is None, "stale-format entry must be invalidated, not decoded"
    assert not _entry_files(str(tmp_path)), "stale entries should be removed"
    # Marker is refreshed to the current version.
    assert (tmp_path / VERSION_FILENAME).read_text(encoding="utf-8").strip() == str(CACHE_FORMAT_VERSION)


def test_empty_unstamped_dir_is_not_an_error(tmp_path):
    """An empty dir with no marker is just a fresh cache: stamp it, no warning-worthy wipe."""
    backend = FileBackend(str(tmp_path))
    backend._ensure_initialized()
    assert (tmp_path / VERSION_FILENAME).read_text(encoding="utf-8").strip() == str(CACHE_FORMAT_VERSION)


def test_clear_preserves_version_marker(tmp_path):
    """clear() removes entries but leaves the format stamp in place."""
    backend = FileBackend(str(tmp_path))
    backend.set("k", {"v": 1})
    backend.get("k")
    backend.clear()
    assert (tmp_path / VERSION_FILENAME).exists()
    assert (tmp_path / VERSION_FILENAME).read_text(encoding="utf-8").strip() == str(CACHE_FORMAT_VERSION)


# -- a cache cleared under a live process -----------------------------------


def test_a_directory_recreated_by_a_live_backend_is_stamped(tmp_path):
    """`cash clear` removes the stamp with the directory; the live process
    recreated the directory for its next write but did not re-stamp it, and
    the next process discarded everything it had written as unknown format."""
    import shutil

    cache = tmp_path / "c"
    live = FileBackend(str(cache))
    live.set("before", {"v": 1})
    live.get("before")  # the write has landed
    shutil.rmtree(cache)  # what `cash clear` does to it
    live.set("after", {"v": 2})
    live.get("after")
    live.shutdown()

    assert (cache / VERSION_FILENAME).read_text(encoding="utf-8").strip() == str(CACHE_FORMAT_VERSION)
    fresh = FileBackend(str(cache))
    assert fresh.get("after")[1] == {"v": 2}, "the next process threw it away"


def test_unstamped_entries_in_the_current_format_are_kept(tmp_path):
    """The entries say what they are; a missing stamp alone is not corruption."""
    b1 = FileBackend(str(tmp_path))
    b1.set("k", {"v": 1})
    b1.get("k")
    b1.shutdown()
    (tmp_path / VERSION_FILENAME).unlink()

    b2 = FileBackend(str(tmp_path))
    assert b2.get("k")[1] == {"v": 1}
    assert (tmp_path / VERSION_FILENAME).read_text(encoding="utf-8").strip() == str(CACHE_FORMAT_VERSION)


def test_unstamped_entries_that_are_not_the_current_format_are_still_wiped(tmp_path):
    """The control: an .entry file without the current magic is not adopted."""
    (tmp_path / f"{'0' * 64}{ENTRY_SUFFIX}").write_bytes(b"CSH1" + b"\x00" * 32)
    b = FileBackend(str(tmp_path))
    b.get("anything")  # initialises
    assert not _entry_files(str(tmp_path))


# -- a cache written by a NEWER cash -----------------------------------------


def _fill_as_a_newer_cash(cache_dir, count: int = 5) -> list[str]:
    """Store *count* entries, then stamp the directory as the next format
    would: what an environment on a newer cash leaves behind."""
    writer = FileBackend(str(cache_dir))
    for i in range(count):
        writer.set(f"k{i}", {"v": i})
    writer.shutdown()
    (cache_dir / VERSION_FILENAME).write_text(str(CACHE_FORMAT_VERSION + 1), encoding="utf-8")
    files = sorted(_entry_files(str(cache_dir)))
    assert len(files) == count, "precondition: the entries are on disk"
    return files


def test_a_cache_stamped_by_a_newer_cash_is_left_alone(tmp_path):
    """Two environments on different cash versions sharing one directory
    wiped each other's results in turn: every open by the older one deleted
    all the newer one had stored. The older one now leaves them, and the
    stamp, as they are."""
    import warnings

    files = _fill_as_a_newer_cash(tmp_path)
    before = {f: os.stat(f).st_mtime_ns for f in files}

    older = FileBackend(str(tmp_path))
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert older.get("k0") == (None, None), "an entry in a format this build cannot trust was served"
        older.set("mine", {"v": 99})
        assert older.get("mine") == (None, None)
        assert older.list_entries() == []
        older.clear()
        older.shutdown()

    assert {f: os.stat(f).st_mtime_ns for f in sorted(_entry_files(str(tmp_path)))} == before
    assert (tmp_path / VERSION_FILENAME).read_text(encoding="utf-8").strip() == str(CACHE_FORMAT_VERSION + 1)
    said = [str(w.message) for w in rec if getattr(w.message, "code", "") == "CACHE-FORMAT-NEWER"]
    assert len(said) == 1, f"expected one notice, got {[str(w.message) for w in rec]}"
    assert str(tmp_path) in said[0]


def test_a_script_on_an_older_cash_runs_uncached_and_says_why(tmp_path):
    """The whole path, in fresh interpreters: 20 results stored, the stamp
    raised, a second script over the same directory. It computes, prints the
    notice on stderr and deletes nothing."""
    import textwrap

    from tests._scripts import run_python

    cache = tmp_path / ".cash"
    (tmp_path / "fill.py").write_text(
        textwrap.dedent(
            """
            import cash
            @cash.cache
            def slow(x):
                return x * 2
            for i in range(20):
                slow(i)
            """
        ),
        encoding="utf-8",
    )
    (tmp_path / "probe.py").write_text(
        textwrap.dedent(
            """
            import cash
            @cash.cache
            def other(x):
                print("computed", x)
                return x
            print("result", other(1))
            """
        ),
        encoding="utf-8",
    )
    run_python("fill.py", cwd=tmp_path)
    before = sorted(_entry_files(str(cache)))
    assert len(before) == 20
    (cache / VERSION_FILENAME).write_text(str(CACHE_FORMAT_VERSION + 1), encoding="utf-8")

    done = run_python("probe.py", cwd=tmp_path)

    assert "computed 1" in done.stdout and "result 1" in done.stdout
    assert sorted(_entry_files(str(cache))) == before, "a newer cash's entries were deleted"
    assert done.stderr.count("CACHE-FORMAT-NEWER") == 1, done.stderr
