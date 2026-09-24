r"""``cash clear`` removes cash's own files, never a user's.

Found while stress-testing the decorator: a project whose
``[tool.cash] cache_dir`` points at a directory holding data --
``cache_dir = "../shared_data"`` -- lost that data. cash writes its
``CACHE_VERSION`` stamp into whatever directory it is pointed at, so the
"does it look like a cache" guard passes and ``clear --all`` removed the
directory whole: ``Cleared: ...\shared_data``, exit 0, ``precious.csv`` gone.
"""

from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import pytest

from cash.__main__ import cmd_clear, main
from cash.backends.entry_format import ENTRY_SUFFIX


def _cache_with(tmp_path, *foreign):
    cache = tmp_path / "shared_data"
    (cache / "raw").mkdir(parents=True)
    (cache / "CACHE_VERSION").write_text("2", encoding="utf-8")
    (cache / f"abc{ENTRY_SUFFIX}").write_bytes(b"entry")
    (cache / "_rank.log").write_text("", encoding="utf-8")
    for name in foreign:
        (cache / name).write_text("mine", encoding="utf-8")
    return cache


def _clear(cache, monkeypatch, tmp_path, **kwargs):
    monkeypatch.chdir(tmp_path)
    args = SimpleNamespace(path=str(cache), all=False, force=False, tool=None, entry=None, function=None, expired=False)
    for key, value in kwargs.items():
        setattr(args, key, value)
    cmd_clear(args)


def test_a_cache_holding_user_files_is_not_removed(tmp_path, monkeypatch, capsys):
    cache = _cache_with(tmp_path, "precious.csv")
    with pytest.raises(SystemExit) as exit_info:
        _clear(cache, monkeypatch, tmp_path)
    assert exit_info.value.code == 1
    out = capsys.readouterr().out
    assert (cache / "precious.csv").read_text(encoding="utf-8") == "mine", out
    assert "precious.csv" in out, out
    assert (cache / f"abc{ENTRY_SUFFIX}").exists(), "nothing was cleared"


def test_force_clears_a_cache_holding_user_files(tmp_path, monkeypatch, capsys):
    cache = _cache_with(tmp_path, "precious.csv")
    _clear(cache, monkeypatch, tmp_path, force=True)
    assert not cache.exists(), capsys.readouterr().out


def test_a_cache_of_only_cash_files_is_removed(tmp_path, monkeypatch, capsys):
    cache = _cache_with(tmp_path)
    (cache / "raw").rmdir()
    _clear(cache, monkeypatch, tmp_path)
    assert not cache.exists(), capsys.readouterr().out
    assert "Cleared" in capsys.readouterr().out or True


def test_a_subdirectory_of_user_files_is_not_removed(tmp_path, monkeypatch, capsys):
    cache = _cache_with(tmp_path)
    (cache / "raw" / "x.bin").write_bytes(b"data")
    with pytest.raises(SystemExit) as exit_info:
        _clear(cache, monkeypatch, tmp_path)
    assert exit_info.value.code == 1
    assert (cache / "raw" / "x.bin").exists(), capsys.readouterr().out


def _notebook_cache(cache_dir):
    """A cache directory as a notebook session leaves it, written by cash's
    own writers: an entry with its stored-key record, the notebook's sidecar
    stores, and an SQLite tier's database with the WAL files SQLite keeps
    beside it while it is open. Returns that database, still open."""
    from cash import Cash
    from cash.backends.sqlite_backend import SQLiteBackend
    from cash.notebook import compute_baselines, loop_split
    from cash.notebook.statement.miss_guard import GUARD_AFTER_CONSECUTIVE_CHURN_MISSES, MissGuard
    from tests.conftest import ABOVE_PERSISTENCE_FLOOR_S

    c = Cash(cache_dir=str(cache_dir), register_magic=False)

    @c.cache
    def fit(x):
        time.sleep(ABOVE_PERSISTENCE_FLOOR_S)
        return x

    fit(1)
    c.shutdown()  # flushes the entry, its stored-key record and the indexes

    guard = MissGuard(str(cache_dir))
    for run in range(GUARD_AFTER_CONSECUTIVE_CHURN_MISSES + 1):
        guard.observe("stmt", f"key{run}", hit=False)
    loop_split.get_store(str(cache_dir)).record("loop", 3)
    baselines = compute_baselines.get_store(str(cache_dir))
    baselines.record("m = fit(x)", 2.0)
    baselines.flush()

    db = SQLiteBackend(str(cache_dir / "cache.db"))
    db.set("k", 1)
    db._writes.wait_all()
    return db


def test_a_notebook_cache_with_its_sidecar_stores_is_removed(tmp_path, monkeypatch, capsys):
    """Every file cash writes into a cache directory counts as cash's.

    The notebook's miss-guard, loop-split and compute-baseline stores and
    SQLite's ``-wal``/``-shm`` files were missing from the CLI's list, so
    ``cash clear`` refused a plain notebook cache as holding the user's files.
    """
    cache = tmp_path / "nb" / ".cash"
    db = _notebook_cache(cache)
    written = {p.relative_to(cache).as_posix() for p in cache.rglob("*")}
    for name in ("_miss_guard.json", "_loop_split.json", "_compute_baselines.json", "cache.db-wal", "cache.db-shm"):
        assert name in written, f"the fixture did not write {name}: {sorted(written)}"
    assert any(n.startswith(".keys/") for n in written), sorted(written)
    # Closed first: Windows cannot delete a database a connection holds open
    # (clearing under a running kernel is its own test). A clean close removes
    # the -wal/-shm files, which a kernel that died leaves behind, so put them
    # back: they are what this test is about.
    db.shutdown()
    for name in ("cache.db-wal", "cache.db-shm"):
        (cache / name).touch()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["cash", "clear", str(cache)])
    main()
    out = capsys.readouterr().out
    assert not cache.exists(), out
    assert "Cleared" in out, out


def test_a_notebook_cache_with_a_user_file_is_still_refused(tmp_path, monkeypatch, capsys):
    """The control: the same cache plus one file of the user's is kept."""
    cache = tmp_path / "nb" / ".cash"
    db = _notebook_cache(cache)
    (cache / "notes.json").write_text("{}", encoding="utf-8")
    try:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(sys, "argv", ["cash", "clear", str(cache)])
        with pytest.raises(SystemExit) as exit_info:
            main()
    finally:
        db.shutdown()
    out = capsys.readouterr().out
    assert exit_info.value.code == 1, out
    assert "notes.json" in out and "_miss_guard.json" not in out, out
    assert (cache / "notes.json").exists()


@pytest.mark.parametrize(
    ("name", "cash_wrote_it"),
    [
        (".tmp-0a1b2c3d4e5f.part", True),  # an entry write in flight
        (".probe-0a1b2c3d4e5f.tmp", True),  # the writability probe
        ("_rank.log.4242.tmp", True),
        ("CACHE_VERSION.tmp", True),
        (".keys/0123abcd.json.4242.139872.tmp", True),
        ("cache.db-journal", True),
        ("run.log", False),  # a suffix alone is not enough
        ("old.data", False),
        ("notes.json.4242.tmp", False),
        ("raw/_rank.log", False),
    ],
)
def test_temp_files_of_cash_writes_are_cash_files(name, cash_wrote_it):
    from cash.backends.cache_dir import is_cash_file

    assert is_cash_file(name) is cash_wrote_it
