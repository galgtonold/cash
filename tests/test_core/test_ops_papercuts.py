"""Round-18 operations papercuts: the things someone running workers trips over.

* `python -m pkg.mod` named its functions `mod.f` and `import pkg.mod` named
  them `pkg.mod.f`: two caches for one function.
* `cash clear` on a live pool left results in the workers' RAM tier, served
  as if nothing had happened.
* A hot entry was modified every few seconds to persist its access stamp;
  in a synced folder that re-uploads the whole file each time.
* `.cash` sat next to the project and went into git with it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import textwrap
import time

import pytest

from cash import Cash, FileBackend

pytestmark = [pytest.mark.core, pytest.mark.timeout(300)]


def test_python_dash_m_and_an_import_share_one_entry(tmp_path):
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "__init__.py").write_text("", encoding="utf-8")
    (pkg / "mod.py").write_text(textwrap.dedent('''
        import sys, time
        import cash

        @cash.cache
        def f(x):
            print("[RUN]", file=sys.stderr)  # @cash:assume-safe
            time.sleep(0.2)  # @cash:assume-safe
            return x * 2

        if __name__ == "__main__":
            print(f(3))
    '''), encoding="utf-8")
    env = {k: v for k, v in os.environ.items() if not k.startswith("CASH_")}
    env.update(CASH_CACHE_DIR=str(tmp_path / ".cash"), PYTHONDONTWRITEBYTECODE="1")
    first = subprocess.run([sys.executable, "-m", "pkg.mod"], cwd=str(tmp_path), env=env,
                           capture_output=True, text=True, timeout=120)
    second = subprocess.run([sys.executable, "-c", "import pkg.mod as m; print(m.f(3))"],
                            cwd=str(tmp_path), env=env, capture_output=True, text=True, timeout=120)
    assert first.stdout.strip() == second.stdout.strip() == "6"
    assert "[RUN]" in first.stderr
    assert "[RUN]" not in second.stderr, "the import did not find the -m run's entry"


@pytest.mark.parametrize("how", ["all", "function"])
def test_a_clear_under_a_running_process_empties_its_ram_tier(tmp_path, monkeypatch, how):
    cache_dir = tmp_path / ".cash"
    c = Cash(cache_dir=str(cache_dir), register_magic=False)
    monkeypatch.setattr(type(c.backend), "_GENERATION_CHECK_EVERY", 0.0)
    calls = []

    @c.cache
    def f(x):
        calls.append(x)
        time.sleep(0.15)          # past the persistence floor, so it reaches disk
        return x * 2

    f(1)
    f(1)
    assert calls == [1]
    if how == "all":
        c.backend.backends[-1]._writes.wait_all()
        shutil.rmtree(cache_dir)                   # what `cash clear --all` does
    else:
        from types import SimpleNamespace

        from cash.__main__ import cmd_clear
        c.backend.backends[-1]._writes.wait_all()
        cmd_clear(SimpleNamespace(path=str(cache_dir), all=False, function="f"))
    f(1)
    assert calls == [1, 1], "the RAM tier served a result cleared from disk"


@pytest.mark.parametrize("how", ["all", "function"])
def test_a_clear_reaches_a_process_that_started_with_no_cache(tmp_path, monkeypatch, how):
    """Round 19: the test above checks on every read, so it always saw the
    stamp the first write created. A real process checks once a second: one
    that started with no cache directory saw no stamp, wrote one, and read the
    clear that followed -- the stamp gone again, or rewritten -- as "still
    new", serving the pre-clear answer from RAM 5 times in 5."""
    cache_dir = tmp_path / ".cash"
    assert not cache_dir.exists()
    c = Cash(cache_dir=str(cache_dir), register_magic=False)
    monkeypatch.setattr(type(c.backend), "_GENERATION_CHECK_EVERY", 3600.0)
    calls = []

    @c.cache
    def f(x):
        calls.append(x)
        time.sleep(0.15)
        return x * 2

    f(1)
    f(1)
    assert calls == [1]
    c.backend.backends[-1]._writes.wait_all()
    if how == "all":
        shutil.rmtree(cache_dir)
    else:
        from types import SimpleNamespace

        from cash.__main__ import cmd_clear
        cmd_clear(SimpleNamespace(path=str(cache_dir), all=False, function="f"))
    monkeypatch.setattr(type(c.backend), "_GENERATION_CHECK_EVERY", 0.0)   # the second has passed
    f(1)
    assert calls == [1, 1], "the RAM tier served a result cleared from disk"


def test_a_clear_during_a_call_begun_inside_the_check_window_is_seen(tmp_path, monkeypatch):
    """Round 20 (r20s4): a job that started cold wrote its first stamp, began
    a long call less than a second later (so no check ran), and `cash clear
    --all` landed during it. The long call's own store re-created the
    directory with a NEW stamp, which the next check took for the one this
    process had written: the cleared value was served for the rest of the run,
    7 times in 10."""
    # Whichever writes into the cleared directory first stamps it: the
    # stored-key record (a stamp this process did not write, always noticed),
    # or the entry's own write -- the case that was not, about 2 runs in 5.
    # Several trials, so the second happens.
    for trial in range(8):
        cache_dir = tmp_path / f"trial{trial}" / ".cash"
        c = Cash(cache_dir=str(cache_dir), register_magic=False)
        monkeypatch.setattr(type(c.backend), "_GENERATION_CHECK_EVERY", 3600.0)
        disk = c.backend.backends[-1]
        calls = []

        @c.cache
        def rate(x):
            calls.append(x)
            time.sleep(0.15)
            return x * 2

        @c.cache
        def long_report(x):
            disk._writes.wait_all()
            shutil.rmtree(cache_dir)             # the operator's clear, mid-call
            time.sleep(0.15)
            return x

        rate(1)
        long_report(1)                           # its store re-creates the directory
        disk._writes.wait_all()
        monkeypatch.setattr(type(c.backend), "_GENERATION_CHECK_EVERY", 0.0)
        rate(1)
        assert calls == [1, 1], f"trial {trial}: the RAM tier served a result cleared from disk"
        c.shutdown()


def test_a_directory_recreated_by_a_sidecar_write_gets_its_gitignore_and_stamp(tmp_path):
    """Round 19: after a clear, the stored-key record re-created the cache
    directory with a bare makedirs, and the entry written next found it there:
    no .gitignore and no format stamp (1 run in 3-6)."""
    cache_dir = tmp_path / ".cash"
    c = Cash(cache_dir=str(cache_dir), register_magic=False)

    @c.cache
    def f(x):
        return x

    f(1)
    c.backend.backends[-1]._writes.wait_all()
    shutil.rmtree(cache_dir)
    c._record_stored_key("mod.f", "mod.f:key", None)
    assert (cache_dir / ".gitignore").read_text(encoding="utf-8").strip().endswith("*")
    assert (cache_dir / "CACHE_VERSION").exists()


def test_the_access_stamp_of_a_hot_entry_is_not_rewritten_on_every_flush(tmp_path, monkeypatch):
    backend = FileBackend(cache_dir=str(tmp_path / ".cash"))
    backend.set("k", 1, {"key": "k"})
    backend._writes.wait_all()
    writes = []
    import cash.backends.file_backend as fb
    real = fb.update_metadata_in_place
    monkeypatch.setattr(fb, "update_metadata_in_place",
                        lambda path, meta: (writes.append(path), real(path, meta))[1])
    for _ in range(3):
        backend.get("k")
        backend._flush_metadata(periodic=True)
    assert len(writes) == 1, "every periodic flush rewrote the entry"
    backend.get("k")
    backend._flush_metadata()                      # shutdown: everything outstanding
    assert len(writes) == 2


def test_a_new_cache_directory_ignores_itself_in_git(tmp_path):
    new = tmp_path / "fresh" / ".cash"
    FileBackend(cache_dir=str(new)).set("k", 1, {"key": "k"})
    assert (new / ".gitignore").read_text(encoding="utf-8").strip().endswith("*")
    existing = tmp_path / "project"
    existing.mkdir()
    FileBackend(cache_dir=str(existing)).set("k", 1, {"key": "k"})
    assert not (existing / ".gitignore").exists(), "wrote a .gitignore into a directory it did not create"
