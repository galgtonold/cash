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

from cash.notebook.file_tracker import FileAccessTracker


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
