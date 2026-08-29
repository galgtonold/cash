"""A cache entry this environment cannot read must be ABSENT, never fatal.

Reported live: opening a notebook in a fresh venv without numpy raised
``ModuleNotFoundError`` out of ``%cash_on`` and killed the cell. The cache held
an entry whose metadata carried ``size = numpy.int64(...)`` -- written by an
environment that had numpy -- and ``FileBackend`` unpickled every metadata
file at startup behind a guard that caught only
``(OSError, pickle.PickleError)``. Startup no longer reads them, but the
eviction path still walks them all, so the guard is what keeps one poisoned
file from taking down a session.

Two independent defects, so two independent sets of tests:

* the READ side must survive an entry it cannot unpickle, whatever the reason;
* the WRITE side must not put a numpy scalar in metadata in the first place.

Either fix alone leaves a real hole: existing caches are already poisoned, and
a future non-builtin scalar would poison new ones.
"""
import pickle

import pytest

from cash.backends.file_backend import FileBackend


class _Exploding:
    """Unpickles only where ``_cas_probe_missing_mod`` is importable, i.e. nowhere.

    Stands in for the real case (a numpy scalar in an environment without
    numpy) without needing to uninstall anything.
    """

    def __reduce__(self):
        return (_missing_module_loader, ())


def _missing_module_loader():  # pragma: no cover - never actually called
    import _cas_probe_missing_mod  # noqa: F401
    return 1


def _poison(cache_dir, key="poisoned"):
    """Write a metadata file that cannot be unpickled in this process."""
    meta = cache_dir / f"{key}.meta"
    meta.write_bytes(pickle.dumps({"key": key, "size": _Exploding()}))
    (cache_dir / f"{key}.data").write_bytes(b"x")
    return meta


def _write_and_settle(cache_dir, key="good", value=None):
    """``set`` the entry and make it DURABLE before another instance reads it.

    ``FileBackend.set`` queues an async write; ``list_entries``/``cleanup``
    drain ``self._writes`` first, which covers "the caller already set() this"
    only when the caller is the SAME instance. These tests deliberately read
    through a FRESH backend — a fresh instance's queue is empty, so it drains
    nothing and the `.meta` file may simply not exist yet.

    That is exactly how these two went red on ubuntu 3.13/3.14 while every
    other job passed: `list_entries()` returned `[]`, missing the GOOD entry,
    not just the poisoned one. Unreproducible in 40 local runs — a runner-speed
    race, not a logic error, which is why it looked like flakiness.

    ``shutdown()`` is the public drain (`_writes.shutdown(wait=True)` then
    `_flush_metadata()`), and it is what a real process does when it finishes.
    """
    backend = FileBackend(cache_dir=str(cache_dir))
    backend.set(key, {"v": 1} if value is None else value)
    backend.shutdown()
    return backend


def test_a_poisoned_entry_does_not_break_startup(tmp_path):
    """Opening a cache must survive a poisoned entry -- one bad file used to
    take down the whole session."""
    _write_and_settle(tmp_path)
    _poison(tmp_path)

    fresh = FileBackend(cache_dir=str(tmp_path))
    assert fresh.get("good")[1] == {"v": 1}, "the readable entry was lost too"


def test_a_poisoned_entry_does_not_break_list_entries(tmp_path):
    """``list_entries`` is what ``%cash_on`` calls to report cache state."""
    _write_and_settle(tmp_path)
    _poison(tmp_path)

    entries = FileBackend(cache_dir=str(tmp_path)).list_entries()
    assert [e.get("key") for e in entries] == ["good"], (
        "the poisoned entry should be skipped, not raise and not be returned"
    )


def test_a_poisoned_entry_reads_as_absent_not_as_an_error(tmp_path):
    backend = FileBackend(cache_dir=str(tmp_path))
    _poison(tmp_path, key="gone")
    assert backend.get_metadata("gone") is None
    assert backend.get("gone") == (None, None)


def test_a_poisoned_entry_does_not_break_cleanup(tmp_path):
    backend = FileBackend(cache_dir=str(tmp_path))
    backend.set("good", {"v": 1})
    _poison(tmp_path)
    assert backend.cleanup_expired(lambda meta: False) == 0


def test_metadata_size_is_a_builtin_int_for_a_numpy_value():
    """The root cause. ``size`` is written into metadata, so a numpy scalar
    there makes the file unreadable wherever numpy is absent."""
    np = pytest.importorskip("numpy")
    from cash.backends.memory_backend import InMemoryBackend

    backend = InMemoryBackend()
    backend.set("arr", np.arange(1000))
    meta, _ = backend.get("arr")
    assert type(meta["size"]) is int, (
        f"size is {type(meta['size'])}, which drags that module into every "
        "future read of this entry's metadata"
    )


def test_metadata_size_is_a_builtin_int_for_a_dataframe():
    """The exact shape that poisoned the real cache: a pandas DataFrame, whose
    ``memory_usage().sum()`` is a ``numpy.int64``."""
    pd = pytest.importorskip("pandas")
    from cash.backends.memory_backend import InMemoryBackend

    backend = InMemoryBackend()
    backend.set("df", pd.DataFrame({"v": range(1000)}))
    meta, _ = backend.get("df")
    assert type(meta["size"]) is int, f"size is {type(meta['size'])}"
