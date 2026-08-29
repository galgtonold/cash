"""``FileAccessTracker`` must not record reads of cash's OWN storage.

A cache hit reads the entry's ``.data`` file to deserialise it, and that read
happens inside the enclosing statement's tracker window. Recording it makes the
statement depend on a cash-internal file.

The consequence is not a stale value -- it is an **unstable lineage**. The
dependency exists on a run where the inner call hit and not on one where it
missed, so a statement's output lineage differs between those two runs.
Anything that chains on that lineage then settles one link per run instead of
immediately. Observed on an 8-iteration loop whose callee mutates a global
(CAS-265's shape), real calls per restart::

    without the guard   7, 6, 5, 4, 3     one iteration settles per run, O(N)
    with the guard      0, 0, 0, 0, 0

and the offending dependency printed as, verbatim:

    .cash/14f9ea2e39789732451365d7201dd60718d7b1ff4a0aa8110738f8952484660d.data

Asserted here against the tracker itself rather than through a notebook,
because whether the dependency reaches persisted metadata depends on which
caching paths are active -- the guard is the invariant, and it holds in every
configuration.

Same class as the ``/proc`` pseudo-filesystem guard: cash's own I/O must never
become a user-visible dependency. That one was found through a stale value,
this one through a cache that would not settle.
"""
import pytest

from cash.notebook.file_tracker import FileAccessTracker, register_cache_dir


@pytest.fixture
def workspace(tmp_path):
    (tmp_path / ".cash").mkdir()
    (tmp_path / ".cash" / "abc123.data").write_bytes(b"cached-value")
    (tmp_path / "_global_cash").mkdir()
    (tmp_path / "_global_cash" / "def456.data").write_bytes(b"cached-value")
    (tmp_path / "real_data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    return tmp_path


def _tracked(workspace, *paths):
    tracker = FileAccessTracker(user_ns={})
    with tracker:
        for p in paths:
            with open(p, "rb"):
                pass
    return {str(f).replace("\\", "/") for f in tracker.get_accessed_files()}


def test_a_read_of_the_local_cache_dir_is_not_tracked(workspace):
    tracked = _tracked(workspace, workspace / ".cash" / "abc123.data")
    assert not [t for t in tracked if "/.cash/" in t], tracked


def test_a_read_of_the_global_cache_dir_is_not_tracked(workspace):
    tracked = _tracked(workspace, workspace / "_global_cash" / "def456.data")
    assert not [t for t in tracked if "/_global_cash/" in t], tracked


def test_a_real_data_file_is_still_tracked(workspace):
    """The guard must not be so broad that it silences genuine dependencies --
    without this, excluding everything would pass the two tests above."""
    tracked = _tracked(workspace, workspace / "real_data.csv")
    assert any(t.endswith("real_data.csv") for t in tracked), tracked


def test_the_two_are_separated_within_one_tracker_window(workspace):
    """The realistic shape: a statement reads user data AND cash reads an entry
    back inside the same window. Only the first is a dependency."""
    tracked = _tracked(
        workspace,
        workspace / "real_data.csv",
        workspace / ".cash" / "abc123.data",
    )
    assert any(t.endswith("real_data.csv") for t in tracked), tracked
    assert not [t for t in tracked if "/.cash/" in t], tracked


class TestACacheDirectoryNotCalledDotCash:
    """The segment guard only recognises `.cash` and `_global_cash` by NAME.

    Any other ``cache_dir`` -- one from configuration, a temp directory, an
    absolute path -- went unguarded, so cash's own entry files became
    dependencies of the user's functions. It stayed invisible while entries
    were renamed into place, because a renamed file is only ever observed at
    its final size. Writing a new entry in place made it visible immediately:
    ``O_CREAT`` leaves the file briefly at zero length and the next check
    reports "size changed", so a fresh process missed on the first call to
    every cached function.

    Backends register their directory now, so the guard does not depend on
    what it is called.
    """

    def test_an_entry_in_an_oddly_named_cache_dir_is_not_tracked(self, tmp_path):
        cache = tmp_path / "my-caches" / "project-a"
        cache.mkdir(parents=True)
        entry = cache / ("a" * 64 + ".entry")
        entry.write_bytes(b"cached")

        assert _tracked(tmp_path, entry), (
            "fixture is wrong: this should be tracked before registering"
        )

        register_cache_dir(str(cache))
        assert not _tracked(tmp_path, entry), (
            "cash's own entry file is still a user-visible dependency"
        )

    def test_user_data_beside_the_cache_is_still_tracked(self, tmp_path):
        """The control, and the reason the guard checks the FILENAME too.

        ``Cash(cache_dir=".")`` is enough to put the cache in a directory that
        also holds the user's data. Ignoring everything under a registered
        directory would drop a real dependency and serve a stale value
        silently -- much worse than the bug the guard exists to prevent, where
        an extra dependency only costs a recompute.
        """
        cache = tmp_path / "workspace"
        cache.mkdir()
        register_cache_dir(str(cache))

        data = cache / "data.csv"
        data.write_text("a,b\n1,2\n")
        entry = cache / ("b" * 64 + ".entry")
        entry.write_bytes(b"cached")

        seen = _tracked(tmp_path, data, entry)
        assert str(data).replace("\\", "/") in seen, (
            f"a user data file living beside the cache was ignored: {seen}"
        )
        assert str(entry).replace("\\", "/") not in seen, (
            f"cash's own entry file was tracked: {seen}"
        )
